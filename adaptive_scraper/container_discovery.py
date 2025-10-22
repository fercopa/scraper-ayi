from collections import Counter, defaultdict

from bs4 import BeautifulSoup, Tag

from core import settings


class ContainerDiscovery:
    """Discovers article containers by finding repeating DOM patterns.

    Works by:
    1. Analyzing DOM structure to find similar elements
    2. Clustering elements with similar structure
    3. Identifying the most likely article containers
    """

    def __init__(
        self: "ContainerDiscovery",
        min_containers: int = settings.DEFAULT_MIN_CONTAINERS,
        max_depth: int = settings.DEFAULT_MAX_DEPTH,
    ) -> None:
        """Initialize ContainerDiscovery.

        Args:
            min_containers: Minimum number of containers to consider valid pattern
            max_depth: Maximum DOM depth to analyze
        """
        self.min_containers = min_containers
        self.max_depth = max_depth

    def discover(self: "ContainerDiscovery", html: str) -> list[Tag]:
        """Discover article containers in HTML.

        Args:
            html: HTML content as string

        Returns:
            List of BeautifulSoup Tag objects representing containers
        """
        soup = BeautifulSoup(html, "html.parser")
        candidates = self._find_candidate_containers(soup)
        if not candidates:
            return []

        # Group by structural similarity
        groups = self._group_by_structure(candidates)
        # Find the best group (largest group of similar elements)
        best_group = self._select_best_group(groups)
        return best_group

    def _find_candidate_containers(self: "ContainerDiscovery", soup: BeautifulSoup) -> list[Tag]:
        """Find all elements that could potentially be article containers."""
        candidates = []

        # Focus on common container tags
        container_tags = ["div", "article", "section", "li"]

        for tag_name in container_tags:
            for element in soup.find_all(tag_name):
                if self._is_valid_candidate(element):
                    candidates.append(element)

        return candidates

    def _is_valid_candidate(self: "ContainerDiscovery", element: Tag) -> bool:
        """Check if element is a valid container candidate.

        Heuristics:
            - Has children (not a leaf node)
            - Contains text content
            - Has links or images
            - Not too shallow (depth > 3)
            - Not too deep (depth < max_depth)
        """
        # Must have children
        if len(list(element.children)) == 0:
            return False

        # Must have some text content
        text = element.get_text(strip=True)
        if len(text) < settings.CONTAINER_MIN_TEXT_LENGTH:
            return False

        # Should have links or images (typical for news articles)
        has_links = element.find("a") is not None
        has_images = element.find("img") is not None

        if not (has_links or has_images):
            return False

        # Check depth (not too shallow, not too deep)
        depth = self._get_depth(element)
        if depth < settings.CONTAINER_MIN_DEPTH or depth > self.max_depth:
            return False
        return True

    def _get_depth(self: "ContainerDiscovery", element: Tag) -> int:
        """Calculate depth of element in DOM tree."""
        depth = 0
        current = element
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def _group_by_structure(self: "ContainerDiscovery", candidates: list[Tag]) -> dict[str, list[Tag]]:
        """Group candidates by structural similarity.

        Creates a "fingerprint" for each element based on:
            - Tag structure of children
            - Number of links, images, text nodes
            - Class names (as features, not fixed selectors)
        """
        groups = defaultdict(list)

        for candidate in candidates:
            fingerprint = self._get_structural_fingerprint(candidate)
            groups[fingerprint].append(candidate)
        return groups

    def _get_structural_fingerprint(self: "ContainerDiscovery", element: Tag) -> str:
        """Create a fingerprint representing the structure of an element.

        This fingerprint should be the same for elements with similar structure,
        even if their content is different.
        """
        features = []

        # Count children by tag type
        child_tags = [child.name for child in element.children if hasattr(child, "name") and child.name]
        tag_counts = Counter(child_tags)
        tag_str = "-".join(f"{k}:{v}" for k, v in sorted(tag_counts.items()) if k)
        features.append(f"tags:{tag_str}")

        # Count specific important elements
        num_links = len(element.find_all("a"))
        num_images = len(element.find_all("img"))
        num_headings = len(element.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))

        features.append(f"links:{num_links}")
        features.append(f"imgs:{num_images}")
        features.append(f"heads:{num_headings}")

        # Depth of nested structure
        max_child_depth = self._get_max_child_depth(element, depth=0, max_depth=5)
        features.append(f"depth:{max_child_depth}")

        # Total children count (binned)
        total_children = len(list(element.descendants))
        child_bin = min(total_children // 10, 10)  # Group into bins of 10
        features.append(f"children_bin:{child_bin}")

        # Create hash of feature combination
        fingerprint = "|".join(features)
        return fingerprint

    def _get_max_child_depth(self: "ContainerDiscovery", element: Tag, depth: int, max_depth: int) -> int:
        """Get maximum depth of children (limited to max_depth)."""
        if depth >= max_depth:
            return depth

        max_child = depth
        for child in element.children:
            if hasattr(child, "children"):
                child_depth = self._get_max_child_depth(child, depth + 1, max_depth)
                max_child = max(max_child, child_depth)
        return max_child

    def _select_best_group(self: "ContainerDiscovery", groups: dict[str, list[Tag]]) -> list[Tag]:
        """Select the best group of containers.

        Prefers:
            1. Groups with more members (more articles = better)
            2. Groups with moderate complexity (not too simple, not too complex)
            3. Groups that appear in sequence (siblings)
        """
        if not groups:
            return []

        # Filter groups with minimum number of containers
        valid_groups = {k: v for k, v in groups.items() if len(v) >= self.min_containers}

        if not valid_groups:
            valid_groups = groups

        scores = {}
        for fingerprint, elements in valid_groups.items():
            score = self._score_group(elements)
            scores[fingerprint] = score
        best_fingerprint = max(scores, key=scores.get)
        return valid_groups[best_fingerprint]

    def _score_group(self: "ContainerDiscovery", elements: list[Tag]) -> float:
        """Score a group of elements.

        Higher score = more likely to be the correct containers
        """
        score = 0.0

        # More elements = better (main signal)
        score += len(elements) * settings.ELEMENT_GROUP_FACTOR

        # Bonus if elements are siblings (appear in sequence)
        sibling_bonus = self._count_sibling_sequences(elements)
        score += sibling_bonus * settings.SIBLING_GROUP_FACTOR

        # Moderate complexity bonus
        avg_text_length = sum(len(el.get_text(strip=True)) for el in elements) / len(elements)
        if 50 < avg_text_length < 500:  # Reasonable article preview length
            score += 10

        # Bonus if elements have images (news articles usually have images)
        elements_with_images = sum(1 for el in elements if el.find("img"))
        image_ratio = elements_with_images / len(elements)
        score += image_ratio * 5

        return score

    def _count_sibling_sequences(self: "ContainerDiscovery", elements: list[Tag]) -> int:
        """Count how many elements are siblings (appear consecutively in DOM)."""
        if len(elements) < 2:
            return 0

        # Group by parent
        by_parent = defaultdict(list)
        for el in elements:
            if el.parent:
                by_parent[id(el.parent)].append(el)

        # Count longest sequence
        max_sequence = 0
        for _, children in by_parent.items():
            if len(children) > max_sequence:
                max_sequence = len(children)

        return max_sequence

    def get_discovery_info(self: "ContainerDiscovery", html: str) -> dict:
        """Get detailed information about the discovery process.

        Useful for debugging and understanding why certain containers were chosen.
        """
        soup = BeautifulSoup(html, "html.parser")
        candidates = self._find_candidate_containers(soup)
        groups = self._group_by_structure(candidates)

        info = {"total_candidates": len(candidates), "num_groups": len(groups), "groups": {}}
        for fingerprint, elements in groups.items():
            score = self._score_group(elements)
            info["groups"][fingerprint] = {
                "count": len(elements),
                "score": score,
                "sample_classes": [el.get("class", []) for el in elements[:3]],
                "sample_text": [el.get_text(strip=True)[:80] for el in elements[:3]],
            }
        return info
