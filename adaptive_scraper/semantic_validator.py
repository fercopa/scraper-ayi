import json
import os
from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from scrapy import Selector
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core import settings


class SemanticValidator:
    """Validates extracted elements using semantic similarity.

    Uses pre-trained sentence-transformers models to:
        1. Encode candidate text into embeddings
        2. Compare with known good examples
        3. Calculate similarity scores
        4. Boost/penalize element scores based on semantic match
    """

    def __init__(
        self: "SemanticValidator",
        model_name: str = settings.DEFAULT_MODEL_NAME,
        cache_dir: str | None = None,
        examples_file: str | None = None,
    ) -> None:
        """Initialize the semantic validator.

        Args:
            model_name: HuggingFace model name for embeddings
                       'all-MiniLM-L6-v2' - Fast, good quality (default)
                       'all-mpnet-base-v2' - Higher quality, slower
            cache_dir: Directory to cache model and embeddings
            examples_file: Path to JSON file with examples (semantic_examples.json)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or ".semantic_cache"

        # Lazy load model (only when needed)
        self._model = None

        # Known good examples (will be populated from training data)
        self.title_examples = []
        self.kicker_examples = []

        # Cached embeddings
        self._title_embeddings = None
        self._kicker_embeddings = None

        # Auto-load examples if provided
        if examples_file and os.path.exists(examples_file):
            self.load_examples_from_json(examples_file)

    @property
    def model(self: "SemanticValidator") -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            print(f"Loading semantic model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("Model loaded")
        return self._model

    def load_examples_from_training_data(self: "SemanticValidator", json_file: str, max_examples: int = 50) -> None:
        """Load good examples from labeled training data.

        Args:
            json_file: Path to JSON file with labeled data
            max_examples: Maximum examples to load per type
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        # Handle both list and single dict formats
        if isinstance(data, dict):
            data = [data]

        print(f"\nLoading examples from {json_file}...")

        for sample in data[:max_examples]:
            try:
                html = sample["html_content"]
                annotations = sample["annotations"]

                soup = BeautifulSoup(html, "html.parser")

                # Get first container using ground truth selector
                container_selector = annotations["container_selector"]
                containers = soup.select(container_selector)

                if not containers:
                    continue

                # Extract title and kicker from first container
                container = containers[0]
                sel = Selector(text=str(container))

                title = sel.css(annotations["title_selector"]).get()
                kicker = sel.css(annotations["kicker_selector"]).get()

                if title:
                    title = title.strip()
                    if len(title) > settings.MINIMUM_TITLE_LENGTH and title not in self.title_examples:
                        self.title_examples.append(title)

                if kicker:
                    kicker = kicker.strip()
                    if len(kicker) > settings.MINIMUM_KICKER_LENGTH and kicker not in self.kicker_examples:
                        self.kicker_examples.append(kicker)

            except Exception as e:
                # Skip samples with errors
                print(f"Error processing sample: {e}")
                continue

        print(f"Loaded {len(self.title_examples)} title examples")
        print(f"Loaded {len(self.kicker_examples)} kicker examples")

        self._generate_embeddings()

    def load_examples_from_json(self: "SemanticValidator", json_file: str) -> None:
        """Load examples from pre-extracted JSON file.

        Args:
            json_file: Path to semantic_examples.json
        """
        print(f"\nLoading examples from {json_file}...")

        with open(json_file, "r") as f:
            data = json.load(f)

        self.title_examples = data.get("titles", [])
        self.kicker_examples = data.get("kickers", [])

        print(f"Loaded {len(self.title_examples)} title examples")
        print(f"Loaded {len(self.kicker_examples)} kicker examples")

        self._generate_embeddings()

    def _generate_embeddings(self: "SemanticValidator") -> None:
        """Generate embeddings for loaded examples."""
        if self.title_examples:
            print("Generating title embeddings...")
            self._title_embeddings = self.model.encode(self.title_examples, show_progress_bar=False)
            print(f"Title embeddings: {self._title_embeddings.shape}")

        if self.kicker_examples:
            print("Generating kicker embeddings...")
            self._kicker_embeddings = self.model.encode(self.kicker_examples, show_progress_bar=False)
            print(f"Kicker embeddings: {self._kicker_embeddings.shape}")

    def validate_title(self: "SemanticValidator", text: str) -> float:
        """Validate if text is semantically similar to known titles.

        Args:
            text: Candidate title text

        Returns:
            Similarity score 0-1 (higher = more similar to known titles)
        """
        if not text or len(text) < settings.MINIMUM_TITLE_LENGTH:
            return 0.0

        if self._title_embeddings is None or len(self.title_examples) == 0:
            # No examples loaded, return neutral score
            return 0.5

        candidate_embedding = self.model.encode([text], show_progress_bar=False)

        similarities = cosine_similarity(candidate_embedding, self._title_embeddings)[0]

        # Use max similarity (most similar known title)
        max_similarity = float(np.max(similarities))

        # Also calculate average similarity (general fit)
        avg_similarity = float(np.mean(similarities))

        # Combine max and average (weighted toward max)
        score = 0.7 * max_similarity + 0.3 * avg_similarity

        return score

    def validate_kicker(self: "SemanticValidator", text: str) -> float:
        """Validate if text is semantically similar to known kickers.

        Args:
            text: Candidate kicker text

        Returns:
            Similarity score 0-1 (higher = more similar to known kickers)
        """
        if not text or len(text) < settings.MINIMUM_KICKER_LENGTH:
            return 0.0

        if self._kicker_embeddings is None or len(self.kicker_examples) == 0:
            return 0.5

        candidate_embedding = self.model.encode([text], show_progress_bar=False)
        similarities = cosine_similarity(candidate_embedding, self._kicker_embeddings)[0]

        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))

        score = 0.7 * max_similarity + 0.3 * avg_similarity

        return score

    def discriminate_title_vs_kicker(self: "SemanticValidator", text: str) -> dict[str, float]:
        """Determine if text is more likely a title or kicker.

        Returns:
            {
                'is_title': float,      # Probability text is a title
                'is_kicker': float,     # Probability text is a kicker
                'confidence': float     # How confident (difference)
            }
        """
        if not text:
            return {"is_title": 0.5, "is_kicker": 0.5, "confidence": 0.0}

        title_score = self.validate_title(text)
        kicker_score = self.validate_kicker(text)

        # Normalize to probabilities
        total = title_score + kicker_score
        if total > 0:
            title_prob = title_score / total
            kicker_prob = kicker_score / total
        else:
            title_prob = 0.5
            kicker_prob = 0.5

        # Confidence is how different they are
        confidence = abs(title_prob - kicker_prob)

        return {
            "is_title": title_prob,
            "is_kicker": kicker_prob,
            "confidence": confidence,
            "title_similarity": title_score,
            "kicker_similarity": kicker_score,
        }

    def get_validation_info(self: "SemanticValidator") -> dict[str, Any]:
        """Get information about loaded examples and model."""
        return {
            "model_name": self.model_name,
            "title_examples_count": len(self.title_examples),
            "kicker_examples_count": len(self.kicker_examples),
            "sample_titles": self.title_examples[:3],
            "sample_kickers": self.kicker_examples[:3],
        }

    def save_cache(self: "SemanticValidator", filepath: str) -> None:
        """Save examples and embeddings to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cache_data = {
            "title_examples": self.title_examples,
            "kicker_examples": self.kicker_examples,
            "title_embeddings": self._title_embeddings.tolist() if self._title_embeddings is not None else None,
            "kicker_embeddings": self._kicker_embeddings.tolist() if self._kicker_embeddings is not None else None,
        }

        with open(filepath, "w") as f:
            json.dump(cache_data, f)

        print(f"Cache saved to {filepath}")

    def load_cache(self: "SemanticValidator", filepath: str) -> bool:
        """Load examples and embeddings from disk."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, "r") as f:
            cache_data = json.load(f)

        self.title_examples = cache_data["title_examples"]
        self.kicker_examples = cache_data["kicker_examples"]

        if cache_data["title_embeddings"]:
            self._title_embeddings = np.array(cache_data["title_embeddings"])

        if cache_data["kicker_embeddings"]:
            self._kicker_embeddings = np.array(cache_data["kicker_embeddings"])

        print(f"Cache loaded from {filepath}")
        print(f"- {len(self.title_examples)} title examples")
        print(f"- {len(self.kicker_examples)} kicker examples")

        return True


#
#
# def test_semantic_validator():
#     """Test the semantic validator"""
#     print("=" * 70)
#     print("SEMANTIC VALIDATOR TEST")
#     print("=" * 70)
#
#     # Create validator with examples file
#     validator = SemanticValidator(examples_file="semantic_examples.json")
#
#     # Test some examples
#     test_cases = [
#         # Real titles
#         ("Canada lawmakers restart push for national sports betting ad rules", "title"),
#         ("Wazdan content now live with NorthStar Gaming in Ontario", "title"),
#         ("MGM Resorts to sell operations of Beau Rivage", "title"),
#         # Real kickers
#         ("Breaking news", "kicker"),
#         ("Exclusive", "kicker"),
#         ("Latest update", "kicker"),
#         # Dates (common false positives)
#         ("2025-10-15", "neither"),
#         ("October 15, 2025", "neither"),
#         # Short text
#         ("Click here", "neither"),
#     ]
#
#     print("\n" + "=" * 70)
#     print("TEST RESULTS")
#     print("=" * 70)
#
#     for text, expected_type in test_cases:
#         result = validator.discriminate_title_vs_kicker(text)
#
#         print(f"\nText: {text[:60]}")
#         print(f"Expected: {expected_type}")
#         print(f"Title probability: {result['is_title']:.3f}")
#         print(f"Kicker probability: {result['is_kicker']:.3f}")
#         print(f"Confidence: {result['confidence']:.3f}")
#
#         if result["is_title"] > 0.6:
#             predicted = "title"
#         elif result["is_kicker"] > 0.6:
#             predicted = "kicker"
#         else:
#             predicted = "uncertain"
#
#         match = "✓" if predicted == expected_type or expected_type == "neither" else "✗"
#         print(f"Predicted: {predicted} {match}")
#
#
# if __name__ == "__main__":
#     test_semantic_validator()
