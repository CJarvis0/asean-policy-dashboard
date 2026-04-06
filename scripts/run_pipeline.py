from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASEAN Policy Dashboard pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "models"],
        default="all",
        help="Choose which stage to run.",
    )
    parser.add_argument(
        "--build-from-raw-json",
        action="store_true",
        help="Rebuild indicator CSVs from raw JSON files before panel assembly.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--skip-ols", action="store_true")
    parser.add_argument("--skip-fixed-effects", action="store_true")
    parser.add_argument("--skip-predictive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from src.pipeline import run_data_pipeline, run_full_pipeline, run_model_pipeline
    except ModuleNotFoundError as exc:
        missing = str(exc).replace("No module named ", "").strip("'")
        print(
            f"Missing dependency: {missing}. Install requirements first with "
            "`pip install -r requirements.txt`."
        )
        return 1

    if args.stage == "data":
        outputs = run_data_pipeline(
            build_from_raw_json=args.build_from_raw_json,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    elif args.stage == "models":
        outputs = run_model_pipeline(
            run_ols=not args.skip_ols,
            run_fixed_effects=not args.skip_fixed_effects,
            run_predictive=not args.skip_predictive,
        )
    else:
        outputs = run_full_pipeline(
            build_from_raw_json=args.build_from_raw_json,
            start_year=args.start_year,
            end_year=args.end_year,
            run_ols=not args.skip_ols,
            run_fixed_effects=not args.skip_fixed_effects,
            run_predictive=not args.skip_predictive,
        )

    print(json.dumps(outputs, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
