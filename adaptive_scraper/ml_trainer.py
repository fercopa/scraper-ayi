import json
import pickle
from pathlib import Path
from typing import Any

from bs4 import Tag
from scrapy import Selector

from .container_discovery import ContainerDiscovery
from .element_scorer import ElementScorer


class MLWeightOptimizer:
    """Optimizes ElementScorer weights using labeled training data.

    Uses grid search to find optimal weight combinations that maximize
    extraction accuracy on the training set.
    """

    def __init__(self: "MLWeightOptimizer", training_data_dir: str = "output"):
        """Initialize the ML weight optimizer.

        Args:
            training_data_dir: Directory containing labeled JSON files
        """
        self.training_data_dir = Path(training_data_dir)
        self.training_samples = []
        self.best_weights = None
        self.best_score = 0.0

    def load_training_data(self: "MLWeightOptimizer") -> int:
        """Load all labeled JSON files from the training data directory.

        Returns:
            Number of samples loaded
        """
        self.training_samples = []

        json_files = list(self.training_data_dir.glob("*2025*.json"))
        print(f"Found {len(json_files)} JSON files in {self.training_data_dir}")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]

                for sample in samples:
                    if "html_content" in sample and "annotations" in sample:
                        self.training_samples.append(sample)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        print(f"Loaded {len(self.training_samples)} training samples")
        return len(self.training_samples)

    def extract_ground_truth(self: "MLWeightOptimizer", sample: dict[str, Any]) -> list[dict[str, str | None]]:
        """Extract ground truth data from a labeled sample.

        Args:
            sample: Sample with html_content and annotations

        Returns:
            List of ground truth articles with title, kicker, image, link
        """
        html = sample["html_content"]
        annotations = sample["annotations"]
        selector = Selector(text=html)

        # Get containers using ground truth selector
        container_selector = annotations["container_selector"]
        containers = selector.css(container_selector)

        ground_truth_articles = []

        for container in containers:
            article = {
                "title": container.css(annotations["title_selector"]).get(),
                "kicker": container.css(annotations["kicker_selector"]).get(),
                "image": container.css(annotations["image_selector"]).get(),
                "link": container.css(annotations["link_selector"]).get(),
            }

            # Only include if at least title exists
            if article["title"]:
                article["title"] = article["title"].strip() if article["title"] else None
                article["kicker"] = article["kicker"].strip() if article["kicker"] else None
                ground_truth_articles.append(article)
        return ground_truth_articles

    def evaluate_weights(self: "MLWeightOptimizer", weights: dict[str, dict[str, float]]) -> float:
        """Evaluate a set of weights on the training data.

        Args:
            weights: Weight configuration for ElementScorer

        Returns:
            Accuracy score (0-1)
        """
        total_fields = 0
        correct_fields = 0

        # Create scorer with these weights
        scorer = ElementScorer(use_semantic_validation=True)
        scorer.weights = weights

        container_discovery = ContainerDiscovery()

        for sample in self.training_samples:
            ground_truth = self.extract_ground_truth(sample)
            html = sample["html_content"]
            containers = container_discovery.discover(html)
            if not ground_truth or not containers:
                continue
            ground_truth_len = len(ground_truth)

            # Extract using current weights
            for i, container in enumerate(containers[:ground_truth_len]):
                elements = scorer.extract_best_elements(container)
                if i >= ground_truth_len:
                    break
                gt = ground_truth[i]
                # Title comparison
                total_fields += 1
                correct_fields += 1 if self._is_valid_field("title", elements, gt) else 0

                # Kicker comparison
                total_fields += 1
                correct_fields += 1 if self._is_valid_field("kicker", elements, gt) else 0

                # Image comparison
                total_fields += 1
                correct_fields += 1 if self._is_valid_field("image", elements, gt) else 0

                # Link comparison
                total_fields += 1
                correct_fields += 1 if self._is_valid_field("link", elements, gt) else 0

        if total_fields == 0:
            return 0.0

        accuracy = correct_fields / total_fields
        return accuracy

    def _is_valid_field(
        self: "MLWeightOptimizer",
        field: str,
        elements_data: dict[str, Tag | None],
        ground_truth: dict[str, str | None],
    ) -> bool:
        """Check if extracted field matches ground truth."""
        element = elements_data.get(field)
        content = ground_truth.get(field)
        if not element or not content:
            return False
        if field == "image":
            extracted_data = element.get("src", "")
            return content == extracted_data
        if field == "link":
            extracted_data = element.get("href", "")
            return content in extracted_data
        extracted_data = element.get_text(strip=True)
        return content in extracted_data or extracted_data in content

    def grid_search_weights(self: "MLWeightOptimizer", quick_search: bool = False) -> dict[str, dict[str, float]]:
        """Use grid search to find optimal weights.

        Args:
            quick_search: If True, use smaller search space for faster results

        Returns:
            Best weight configuration
        """
        print("\n" + "=" * 70)
        print("GRID SEARCH FOR OPTIMAL WEIGHTS")
        print("=" * 70)

        if quick_search:
            # Quick search: Test fewer combinations
            position_weights = [0.05, 0.10, 0.15]
            semantic_weights = [0.20, 0.25, 0.30]
            tag_weights = [0.20, 0.25, 0.30]
        else:
            # Full search: More granular
            position_weights = [0.05, 0.10, 0.15, 0.20]
            semantic_weights = [0.15, 0.20, 0.25, 0.30, 0.35]
            tag_weights = [0.15, 0.20, 0.25, 0.30]

        best_accuracy = 0.0
        best_config = None
        total_tested = 0

        # Test different weight combinations
        for pos_w in position_weights:
            for sem_w in semantic_weights:
                for tag_w in tag_weights:
                    # Ensure weights sum to reasonable range
                    remaining = 1.0 - pos_w - sem_w - tag_w
                    if remaining < 0.1 or remaining > 0.5:
                        continue

                    # Distribute remaining weight
                    length_w = remaining * 0.6
                    link_w = remaining * 0.4

                    weights = {
                        "title": {
                            "tag_score": tag_w,
                            "position_score": pos_w,
                            "length_score": length_w,
                            "link_score": link_w,
                            "semantic_score": sem_w,
                        },
                        "kicker": {
                            "tag_score": tag_w * 0.8,
                            "position_score": pos_w,
                            "length_score": length_w,
                            "style_score": sem_w,
                            "proximity_score": remaining * 0.4,
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

                    accuracy = self.evaluate_weights(weights)
                    total_tested += 1

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_config = weights
                        print(f"\nNew best: {accuracy:.3f} (pos={pos_w}, sem={sem_w}, tag={tag_w})")

                    if total_tested % 10 == 0:
                        print(f"Tested {total_tested} combinations, best so far: {best_accuracy:.3f}")

        print(f"\n{'=' * 70}")
        print("SEARCH COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total combinations tested: {total_tested}")
        print(f"Best accuracy: {best_accuracy:.3f}")

        self.best_weights = best_config
        self.best_score = best_accuracy

        return best_config

    def save_weights(self: "MLWeightOptimizer", filename: str = "trained_models/optimal_weights.pkl") -> None:
        """Save the best weights to a file.

        Args:
            filename: Path to save the weights
        """
        if self.best_weights is None:
            raise ValueError("No weights to save. Run grid_search_weights() first.")

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "weights": self.best_weights,
                    "score": self.best_score,
                },
                f,
            )

        print(f"\nWeights saved to {output_path}")

        # Also save as JSON for human readability
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "weights": self.best_weights,
                    "score": self.best_score,
                },
                f,
                indent=2,
            )

        print(f"Weights also saved as JSON to {json_path}")

    def load_weights(
        self: "MLWeightOptimizer",
        filename: str = "trained_models/optimal_weights.pkl",
    ) -> dict[str, dict[str, float]]:
        """Load previously saved weights.

        Args:
            filename: Path to the weights file

        Returns:
            Weight configuration
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.best_weights = data["weights"]
        self.best_score = data["score"]

        print(f"Loaded weights from {filename} (score: {self.best_score:.3f})")

        return self.best_weights
