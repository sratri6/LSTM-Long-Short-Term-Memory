import argparse
import sys
import json

import numpy as np
from tensorflow.keras.models import load_model

from evaluate import evaluate_model


def main():
    p = argparse.ArgumentParser(description="Load a saved Keras model and numpy test data, run evaluate_model, and optionally save results.")
    p.add_argument("--model", required=True, help="Path to saved Keras model (.h5 file or SavedModel directory)")
    p.add_argument("--x_test", required=True, help="Path to X_test numpy .npy file")
    p.add_argument("--y_test", required=True, help="Path to y_test numpy .npy file")
    p.add_argument("--output", help="Optional path to write JSON results (loss and accuracy)")
    args = p.parse_args()

    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model from {args.model}: {e}", file=sys.stderr)
        sys.exit(2)

    # Load test data
    try:
        X_test = np.load(args.x_test)
        y_test = np.load(args.y_test)
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(3)

    # Run evaluation
    try:
        loss, accuracy = evaluate_model(model, X_test, y_test)
    except Exception as e:
        # If evaluate_model raises (for example model.evaluate returns different metrics), show error
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(4)

    # Optionally save results to JSON
    if args.output:
        try:
            results = {"loss": float(loss), "accuracy": float(accuracy)}
            with open(args.output, "w") as fh:
                json.dump(results, fh)
            print(f"Saved numeric results to {args.output}")
        except Exception as e:
            print(f"Failed to write output file {args.output}: {e}", file=sys.stderr)
            sys.exit(5)


if __name__ == "__main__":
    main()
