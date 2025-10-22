import os
import pickle
import re

from bs4 import Tag

from core import settings

from .semantic_validator import SemanticValidator


class ElementScorer:
    """Scores elements within a container to identify article components.

    Uses heuristics based on:
        - Position in container
        - Text length and content
        - HTML tag type
        - Visual properties (inferred from HTML)
        - Link patterns
        - Semantic similarity (optional, with SemanticValidator)
    """

    def __init__(
        self: "ElementScorer",
        use_semantic_validation: bool = True,
        use_ml_weights: bool = False,
        model_path: str = "trained_models/optimal_weights.pkl",
        semantic_examples_file: str = "semantic_examples.json",
    ) -> None:
        """Initialize element scorer with default weights.

        Args:
            use_semantic_validation: Enable semantic validation using embeddings
            use_ml_weights: Load ML-optimized weights from trained_models/
            model_path: Path to ML weights file
            semantic_examples_file: Path to semantic examples JSON
        """
        self.use_semantic_validation = use_semantic_validation
        self._semantic_validator = None

        # Weights for different features (optimized for semantic validation)
        self.weights = {
            "title": {
                "tag_score": 0.25,  # Reduced from 0.3 (tag type less critical)
                "position_score": 0.10,  # Reduced from 0.2 (avoid position bias)
                "length_score": 0.25,  # Increased from 0.2 (longer = title)
                "link_score": 0.15,  # Keep same (titles often links)
                "semantic_score": 0.25,  # Increased from 0.15 (semantic validation!)
            },
            "kicker": {
                "tag_score": 0.20,  # Reduced from 0.25
                "position_score": 0.10,  # Reduced from 0.20 (avoid position bias)
                "length_score": 0.25,  # Keep same (kickers shorter)
                "style_score": 0.25,  # Increased from 0.15 (semantic validation!)
                "proximity_score": 0.20,  # Increased from 0.15 (near title)
            },
            "image": {
                "tag_score": 0.4,
                "position_score": 0.2,
                "size_score": 0.2,
                "alt_score": 0.2,
            },
            "link": {
                "href_score": 0.4,
                "position_score": 0.2,
                "wraps_content": 0.4,
            },
        }

        if use_ml_weights:
            self._load_ml_weights(model_path)

    def _load_ml_weights(self: "ElementScorer", model_path: str) -> None:
        """Load ML-optimized weights from trained_models."""
        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                    self.weights = data["weights"]
                    print(f"Loaded ML-optimized weights (accuracy: {data['score']:.3f})")
            else:
                print(f"Warning: {model_path} not found, using default weights")
        except Exception as e:
            print(f"Warning: Could not load ML weights: {e}")

    @property
    def semantic_validator(self: "ElementScorer") -> SemanticValidator:
        """Lazy load semantic validator."""
        if self.use_semantic_validation and self._semantic_validator is None:
            try:
                if os.path.exists(self.semantic_examples_file):
                    self._semantic_validator = SemanticValidator(examples_file=self.semantic_examples_file)
                else:
                    print(f"Warning: {self.semantic_examples_file} not found, semantic validation disabled")
                    self.use_semantic_validation = False
            except Exception as e:
                print(f"Warning: Could not load semantic validator: {e}")
                self.use_semantic_validation = False
        return self._semantic_validator

    def score_title_candidates(self: "ElementScorer", container: Tag) -> list[tuple[Tag, float]]:
        """Score all potential title elements in container.

        Returns:
            List of (element, score) tuples, sorted by score (highest first)
        """
        candidates = []

        # Find all text-containing elements
        for element in container.descendants:
            if not isinstance(element, Tag):
                continue

            text = element.get_text(strip=True)
            if not text or len(text) < settings.MINIMUM_TITLE_LENGTH:
                continue

            # Skip if element contains other candidates (want most specific)
            if self._has_text_children(element):
                continue

            score = self._score_title(element, container)
            if score > settings.MINIMUM_THRESHOLD_SCORE:
                candidates.append((element, score))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _score_title(self: "ElementScorer", element: Tag, container: Tag) -> float:
        """Score an element as a potential title."""
        scores = {}

        tag_hierarchy = {"h1": 1.0, "h2": 0.9, "h3": 0.8, "h4": 0.7, "a": 0.6, "div": 0.4, "span": 0.3, "p": 0.2}
        scores["tag_score"] = tag_hierarchy.get(element.name, 0.1)

        # Position score (higher in container = better)
        scores["position_score"] = self._get_position_score(element, container)

        # Length score (30-200 chars is optimal for title)
        text = element.get_text(strip=True)
        length = len(text)
        if 30 <= length <= 200:
            scores["length_score"] = 1.0
        elif 20 <= length <= 250:
            scores["length_score"] = 0.7
        elif length < 20:
            scores["length_score"] = 0.3
        else:  # Too long
            scores["length_score"] = max(0, 1.0 - (length - 200) / 300)

        # Link score (titles are often links)
        if element.name == "a" or element.find_parent("a"):
            scores["link_score"] = 1.0
        else:
            scores["link_score"] = 0.3

        # Semantic score (capitalization, no numbers at start, etc.)
        scores["semantic_score"] = self._score_title_semantics(text)

        # Weighted average
        weights = self.weights["title"]
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score

    def _score_title_semantics(self: "ElementScorer", text: str) -> float:
        """Score text based on title-like characteristics + semantic validation."""
        if self.use_semantic_validation and self.semantic_validator:
            try:
                result = self.semantic_validator.discriminate_title_vs_kicker(text)
                return result["title_similarity"]
            except Exception as e:
                print(f"Warning: Semantic validation failed: {e}")

        # Fallback: Heuristic scoring
        score = 0.5  # Base score

        # Bonus if first letter is capitalized
        if text and text[0].isupper():
            score += 0.2

        # Penalty if starts with number
        if text and text[0].isdigit():
            score -= 0.3

        # Bonus if doesn't have too many special characters
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
        if special_chars / max(len(text), 1) < 0.1:
            score += 0.2

        # Penalty if all caps (more like kicker)
        if text.isupper() and len(text) > 10:
            score -= 0.4
        return max(0, min(1, score))

    def score_kicker_candidates(
        self: "ElementScorer",
        container: Tag,
        title_element: Tag | None = None,
    ) -> list[tuple[Tag, float]]:
        """Score all potential kicker/subtitle elements.

        Args:
            container: Container element
            title_element: Optional title element to measure proximity
        """
        candidates = []

        for element in container.descendants:
            if not isinstance(element, Tag):
                continue

            text = element.get_text(strip=True)
            if not text or len(text) < settings.MINIMUM_KICKER_LENGTH:
                continue

            # Skip if same as title
            if title_element and element == title_element:
                continue

            # Skip if contains other candidates
            if self._has_text_children(element):
                continue

            score = self._score_kicker(element, container, title_element)
            if score > settings.MINIMUM_THRESHOLD_SCORE:
                candidates.append((element, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _score_kicker(self: "ElementScorer", element: Tag, container: Tag, title_element: Tag | None) -> float:
        """Score element as potential kicker."""
        scores = {}

        # Tag score (span, div, p are common for kickers)
        tag_hierarchy = {"span": 0.8, "div": 0.7, "p": 0.6, "a": 0.5}
        scores["tag_score"] = tag_hierarchy.get(element.name, 0.3)

        scores["position_score"] = self._get_position_score(element, container)

        # Length score (kickers are shorter than titles: 10-100 chars)
        text = element.get_text(strip=True)
        length = len(text)
        if 10 <= length <= 100:
            scores["length_score"] = 1.0
        elif 5 <= length <= 150:
            scores["length_score"] = 0.6
        else:
            scores["length_score"] = 0.2

        # Style score (uppercase, bold common for kickers) + semantic validation
        style_score = 0.5

        # Use semantic validation if available
        if self.use_semantic_validation and self.semantic_validator:
            try:
                result = self.semantic_validator.discriminate_title_vs_kicker(text)
                # Use kicker similarity as bonus
                style_score = 0.3 * style_score + 0.7 * result["kicker_similarity"]
            except Exception:
                print("Warning: Semantic validation failed for kicker")

        # Fallback heuristics
        if text.isupper():
            style_score += 0.3
        # Check for bold-like class names
        classes = element.get("class", [])
        if any("bold" in str(c).lower() or "strong" in str(c).lower() for c in classes):
            style_score += 0.2
        scores["style_score"] = min(1.0, style_score)

        # Proximity score (close to title is better)
        if title_element:
            scores["proximity_score"] = self._get_proximity_score(element, title_element)
        else:
            scores["proximity_score"] = 0.5

        weights = self.weights["kicker"]
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score

    def score_image_candidates(self: "ElementScorer", container: Tag) -> list[tuple[Tag, float]]:
        """Score all potential image elements."""
        candidates = []

        for img in container.find_all("img"):
            score = self._score_image(img, container)
            if score > settings.MINIMUM_THRESHOLD_SCORE:
                candidates.append((img, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _score_image(self: "ElementScorer", img: Tag, container: Tag) -> float:
        """Score image element."""
        scores = {}

        # Tag score (it's an img, so high base score)
        scores["tag_score"] = 1.0
        scores["position_score"] = self._get_position_score(img, container)

        # Size score (inferred from src or attributes)
        size_score = settings.DEFAULT_IMAGE_SIZE_SCORE
        src = img.get("src", "")

        # Penalty for icons/logos (small images)
        if "icon" in src.lower() or "logo" in src.lower():
            size_score = 0.1
            # Bonus for high-res indicators
        elif any(x in src.lower() for x in ["1200", "800", "large", "full"]):
            size_score = 1.0

        scores["size_score"] = size_score

        # Alt text score
        alt = img.get("alt", "")
        if alt and len(alt) > 10:
            scores["alt_score"] = 1.0
        elif alt:
            scores["alt_score"] = 0.5
        else:
            scores["alt_score"] = 0.2

        weights = self.weights["image"]
        total_score = sum(scores[k] * weights[k] for k in scores)

        return total_score

    def score_link_candidates(
        self: "ElementScorer",
        container: Tag,
        title_element: Tag | None = None,
        image_element: Tag | None = None,
    ) -> list[tuple[Tag, float]]:
        """Score all potential article links."""
        candidates = []

        for link in container.find_all("a"):
            score = self._score_link(link, container, title_element, image_element)
            if score > 0.3:
                candidates.append((link, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _score_link(
        self: "ElementScorer",
        link: Tag,
        container: Tag,
        title_element: Tag | None,
        image_element: Tag | None,
    ) -> float:
        """Score link element."""
        scores = {}

        # href score (must point to article)
        href = link.get("href", "")
        href_score = settings.DEFAULT_LINK_SCORE

        # High score for article-like URLs
        if any(x in href.lower() for x in ["/news/", "/article/", "/post/", "/story/"]):
            href_score = 1.0
            # Penalty for non-article links
        elif any(x in href.lower() for x in ["facebook", "twitter", "linkedin", "#"]):
            href_score = 0.1
            # Bonus for relative URLs (likely internal article)
        elif href.startswith("/") and len(href) > 5:
            href_score = 0.9

        scores["href_score"] = href_score
        scores["position_score"] = self._get_position_score(link, container)

        # Wraps content score
        wraps_score = 0.0
        if title_element and link in title_element.parents:
            wraps_score = 1.0
        elif title_element and title_element in link.descendants:
            wraps_score = 1.0
        elif image_element and image_element in link.descendants:
            wraps_score = 0.8

        scores["wraps_content"] = wraps_score
        weights = self.weights["link"]
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score

    def _get_position_score(self: "ElementScorer", element: Tag, container: Tag) -> float:
        """Score based on position in container.

        Earlier elements get higher scores (title/kicker usually at top).
        """
        try:
            all_elements = list(container.descendants)
            if element not in all_elements:
                return 0.5

            position = all_elements.index(element)
            total = len(all_elements)

            # Normalize to 0-1, with earlier = higher
            # First 20% of elements get score 0.8-1.0
            # Next 30% get 0.5-0.8
            # Rest get 0.2-0.5
            relative_pos = position / max(total, 1)

            if relative_pos < 0.2:
                score = 1.0 - (relative_pos / 0.2) * 0.2  # 0.8-1.0
            elif relative_pos < 0.5:
                score = 0.8 - ((relative_pos - 0.2) / 0.3) * 0.3  # 0.5-0.8
            else:
                score = 0.5 - ((relative_pos - 0.5) / 0.5) * 0.3  # 0.2-0.5
            return max(0.1, score)
        except (ValueError, ZeroDivisionError):
            return 0.5

    def _get_proximity_score(self: "ElementScorer", element1: Tag, element2: Tag) -> float:
        """Score based on proximity of two elements.

        Closer elements get higher scores.
        """
        try:
            parent = element1.find_parent()
            if not parent:
                return 0.5

            siblings = list(parent.children)
            siblings = [s for s in siblings if isinstance(s, Tag)]

            if element1 not in siblings or element2 not in siblings:
                # Not siblings, check if nested
                if element2 in element1.parents or element1 in element2.parents:
                    return 1.0
                return 0.3

            idx1 = siblings.index(element1)
            idx2 = siblings.index(element2)
            distance = abs(idx1 - idx2)

            # Adjacent = 1.0, 1 apart = 0.8, 2 apart = 0.6, etc.
            score = max(0.2, 1.0 - (distance * 0.2))
            return score
        except (ValueError, AttributeError):
            return 0.5

    def _has_text_children(self: "ElementScorer", element: Tag) -> bool:
        """Check if element has children with text (not a leaf text node)."""
        for child in element.children:
            if isinstance(child, Tag):
                child_text = child.get_text(strip=True)
                if child_text and len(child_text) > settings.MINIMUM_CHILD_TEXT_LENGTH:
                    return True
        return False

    def extract_best_elements(self: "ElementScorer", container: Tag) -> dict[str, Tag | None]:
        """Extract the best title, kicker, image, and link from a container.

        Returns:
            Dictionary with 'title', 'kicker', 'image', 'link' keys
        """
        result = {
            "title": None,
            "kicker": None,
            "image": None,
            "link": None,
        }
        title_candidates = self.score_title_candidates(container)
        if title_candidates:
            result["title"] = title_candidates[0][0]

        kicker_candidates = self.score_kicker_candidates(container, result["title"])
        if kicker_candidates:
            result["kicker"] = kicker_candidates[0][0]

        image_candidates = self.score_image_candidates(container)
        if image_candidates:
            result["image"] = image_candidates[0][0]

        link_candidates = self.score_link_candidates(container, result["title"], result["image"])
        if link_candidates:
            result["link"] = link_candidates[0][0]

        return result
