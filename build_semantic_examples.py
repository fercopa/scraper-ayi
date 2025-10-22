from adaptive_scraper.semantic_validator import SemanticValidator


def main():
    """Build semantic examples database from all training data."""
    print("=" * 70)
    print("BUILDING SEMANTIC EXAMPLES DATABASE")
    print("=" * 70)

    validator = SemanticValidator()

    # Load from all JSON files
    json_files = [
        "output/extracted_data_20251018.13.14.39.json",  # 420 samples
        "output/data_result_20251018.22.24.11.json",  # 1 sample
        "output/data_result_20251018.22.32.37.json",  # 1 sample
    ]

    for json_file in json_files:
        try:
            validator.load_examples_from_training_data(json_file, max_examples=200)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    info = validator.get_validation_info()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {info['model_name']}")
    print(f"Title examples: {info['title_examples_count']}")
    print(f"Kicker examples: {info['kicker_examples_count']}")

    print("\nSample titles:")
    for title in info["sample_titles"]:
        print(f"  - {title[:80]}")

    print("\nSample kickers:")
    for kicker in info["sample_kickers"]:
        print(f"  - {kicker[:80]}")

    # Save cache
    validator.save_cache(".semantic_cache/embeddings.json")

    # Test validation
    print("\n" + "=" * 70)
    print("TEST VALIDATION")
    print("=" * 70)

    test_texts = [
        "Playson partners with High Flyer Casino to expand Ontario presence",
        "Breaking news",
        "2025-10-15",
        "Latest developments in gaming industry",
    ]

    for text in test_texts:
        result = validator.discriminate_title_vs_kicker(text)
        print(f"\nText: {text}")
        print(f"  Title sim: {result['title_similarity']:.3f}")
        print(f"  Kicker sim: {result['kicker_similarity']:.3f}")
        print(f"  Is title: {result['is_title']:.3f}")
        print(f"  Is kicker: {result['is_kicker']:.3f}")


if __name__ == "__main__":
    main()
