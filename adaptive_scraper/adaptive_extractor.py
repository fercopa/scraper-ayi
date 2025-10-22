from typing import Any

from bs4 import Tag

from core import settings
from utils import format_candidate

from .container_discovery import ContainerDiscovery
from .element_scorer import ElementScorer


class AdaptiveExtractor:
    """AI-powered adaptive news extractor.

    Automatically discovers article containers and extracts Title, Kicker,
    Image, and Link without relying on fixed CSS selectors.

    Features:
    - Adapts to HTML structure changes
    - Works across different site layouts
    - No manual selector configuration needed
    """

    def __init__(
        self: "AdaptiveExtractor",
        min_containers: int = settings.DEFAULT_MIN_CONTAINERS,
        confidence_threshold: float = settings.DEFAULT_CONFIDENCE_THRESHOLD,
        use_ml_weights: bool = False,
    ) -> None:
        """Initialize the AdaptiveExtractor.

        Args:
            min_containers: Minimum containers to find valid pattern
            confidence_threshold: Minimum confidence score for extraction
            use_ml_weights: Load ML-optimized weights from a trained model
        """
        self.container_discovery = ContainerDiscovery(min_containers=min_containers)
        self.element_scorer = ElementScorer(use_ml_weights=use_ml_weights)
        self.confidence_threshold = confidence_threshold

    def extract(self: "AdaptiveExtractor", html: str, url: str | None = None) -> list[dict[str, any]]:
        """Extract articles from HTML.

        Args:
            html: HTML content as string
            url: Optional URL for context

        Returns:
            List of dictionaries with extracted article data:
            [
                {
                    'title': str,
                    'kicker': str,
                    'image': str,
                    'link': str,
                    'confidence': float,  # Overall confidence score
                    'scores': dict,  # Individual element scores
                },
                ...
            ]
        """
        containers = self.container_discovery.discover(html)
        if not containers:
            return []

        articles = []
        for i, container in enumerate(containers):
            article = self._extract_from_container(container)

            if article and article["confidence"] >= self.confidence_threshold:
                article["container_index"] = i
                articles.append(article)
        return articles

    def _extract_from_container(self: "AdaptiveExtractor", container: Tag) -> dict[str, Any] | None:
        """Extract article data from a single container.

        Returns:
            Dictionary with extracted data or None if extraction failed
        """
        elements = self.element_scorer.extract_best_elements(container)

        # Calculate individual scores
        scores = {}
        article_data = {}

        scores["title"], article_data["title"] = self._get_title_info(elements["title"], container)
        scores["kicker"], article_data["kicker"] = self._get_kicker_info(
            elements["kicker"], container, elements["title"]
        )
        scores["image"], article_data["image"] = self._get_image_info(elements["image"], container)
        scores["link"], article_data["link"] = self._get_link_info(
            elements["link"], container, elements["title"], elements["image"]
        )

        # Calculate overall confidence
        weights = {
            "title": settings.TITLE_WEIGHT,
            "kicker": settings.KICKER_WEIGHT,
            "image": settings.IMAGE_WEIGHT,
            "link": settings.LINK_WEIGHT,
        }
        confidence = sum(scores.get(k, 0) * weights[k] for k in weights)

        # Must have at least title or link to be valid
        if not article_data["title"] and not article_data["link"]:
            return None

        return {
            **article_data,
            "confidence": confidence,
            "scores": scores,
        }

    def _get_title_info(self: "AdaptiveExtractor", element: Tag, container: Tag) -> tuple[float, str | None]:
        """Extract title and its score from container.

        Returns:
            (title_score, title_text)
        """
        if not element:
            return (0.0, None)
        title_candidates = self.element_scorer.score_title_candidates(container)
        title_score = title_candidates[0][1] if title_candidates else 0.0
        return (title_score, element.get_text(strip=True))

    def _get_kicker_info(
        self: "AdaptiveExtractor",
        element: Tag,
        container: Tag,
        title_element: Tag,
    ) -> tuple[float, str | None]:
        """Extract kicker and its score from container.

        Returns:
            (kicker_score, kicker_text)
        """
        if not element:
            return (0.0, None)
        kicker_candidates = self.element_scorer.score_kicker_candidates(container, title_element)
        kicker_score = kicker_candidates[0][1] if kicker_candidates else 0.0
        return (kicker_score, element.get_text(strip=True, separator=" "))

    def _get_image_info(self: "AdaptiveExtractor", element: Tag, container: Tag) -> tuple[float, str | None]:
        """Extract image URL and its score from container.

        Returns:
            (image_score, image_url)
        """
        if not element:
            return (0.0, None)
        image_candidates = self.element_scorer.score_image_candidates(container)
        image_score = image_candidates[0][1] if image_candidates else 0.0
        return (image_score, element.get("src", ""))

    def _get_link_info(
        self: "AdaptiveExtractor",
        element: Tag,
        container: Tag,
        title_element: Tag,
        image_element: Tag,
    ) -> tuple[float, str | None]:
        """Extract link URL and its score from container.

        Returns:
            (link_score, link_url)
        """
        if not element:
            return (0.0, None)
        link_candidates = self.element_scorer.score_link_candidates(container, title_element, image_element)
        link_score = link_candidates[0][1] if link_candidates else 0.0
        return (link_score, element.get("href", ""))

    def extract_with_details(self: "AdaptiveExtractor", html: str, url: str | None = None) -> dict[str, Any]:
        """Extract articles with detailed debugging information.

        Useful for understanding how the extractor works and tuning parameters.

        Returns:
            {
                'articles': List[Dict],
                'discovery_info': Dict,
                'container_count': int,
                'total_candidates': int,
            }
        """
        discovery_info = self.container_discovery.get_discovery_info(html)
        articles = self.extract(html, url)
        containers = self.container_discovery.discover(html)

        return {
            "articles": articles,
            "discovery_info": discovery_info,
            "container_count": len(containers),
            "total_candidates": discovery_info["total_candidates"],
            "extracted_count": len(articles),
        }

    def get_element_candidates_details(
        self: "AdaptiveExtractor",
        html: str,
        container_index: int = 0,
    ) -> dict[str, Any]:
        """Get detailed scores for all candidates in a specific container.

        Useful for debugging and understanding scoring.

        Args:
            html: HTML content
            container_index: Index of container to analyze

        Returns:
            Dictionary with candidate scores for title, kicker, image, link
        """
        containers = self.container_discovery.discover(html)

        if container_index >= len(containers):
            return {"error": f"Container index {container_index} out of range"}

        container = containers[container_index]

        # Get all candidates with scores
        title_candidates = self.element_scorer.score_title_candidates(container)
        kicker_candidates = self.element_scorer.score_kicker_candidates(
            container, title_candidates[0][0] if title_candidates else None
        )
        image_candidates = self.element_scorer.score_image_candidates(container)
        link_candidates = self.element_scorer.score_link_candidates(
            container,
            title_candidates[0][0] if title_candidates else None,
            image_candidates[0][0] if image_candidates else None,
        )
        return {
            "container_index": container_index,
            "title_candidates": [format_candidate(e, s) for e, s in title_candidates[:5]],
            "kicker_candidates": [format_candidate(e, s) for e, s in kicker_candidates[:5]],
            "image_candidates": [format_candidate(e, s) for e, s in image_candidates[:5]],
            "link_candidates": [format_candidate(e, s) for e, s in link_candidates[:5]],
        }
