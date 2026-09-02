"""Train the recommended late-state CE recipe: 20K development or 50K reference."""

import argparse

from looping.exp_health_methods import train as train_reference
from looping.exp_stay_solved import train as train_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("development", "reference"), default="development")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--run-name", required=True, help="Use a new name for a new experiment; reuse it to resume")
    parser.add_argument("--output-dir", default="runs/training")
    args = parser.parse_args()
    trainer = train_reference if args.preset == "reference" else train_development
    trainer(output_dir=args.output_dir, arm="late_state_ce", run_name=args.run_name, random_seed=args.seed)


if __name__ == "__main__":
    main()
