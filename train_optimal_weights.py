import json

from adaptive_scraper.ml_trainer import MLWeightOptimizer


def main():
    """Train Optimal Weights for Adaptive Scraper.

    Uses ML to learn the best weight combinations for ElementScorer.
    """
    choice = input("Run [Q]uick search or [F]ull search (slow)? [Q/F]: ").strip().upper()
    quick_search = choice != "F"
    if quick_search:
        print("\nRunning QUICK search (faster results)")
    else:
        print("\nRunning FULL search (best accuracy)")

    optimizer = MLWeightOptimizer()

    print("\n" + "-" * 70)
    print("Step 1: Loading training data")
    print("-" * 70)
    num_samples = optimizer.load_training_data()

    if num_samples == 0:
        print("Error: No training samples found in output/")
        return

    print(f"Loaded {num_samples} samples")

    print("\n" + "-" * 70)
    print("Step 2: Grid search for optimal weights")
    print("-" * 70)
    print("This may take several minutes...")

    best_weights = optimizer.grid_search_weights(quick_search=quick_search)

    # Print results
    print("\n" + "-" * 70)
    print("Step 3: Best weights found")
    print("-" * 70)
    print(json.dumps(best_weights, indent=2))

    # Save
    print("\n" + "-" * 70)
    print("Step 4: Saving weights")
    print("-" * 70)
    optimizer.save_weights()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Weights saved to:")
    print("  - trained_models/optimal_weights.pkl (for Python)")
    print("  - trained_models/optimal_weights.json (human-readable)")
    print()


if __name__ == "__main__":
    main()
