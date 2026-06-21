from __future__ import annotations

import argparse
import json

from language_machine.profile_step import StepProfileConfig, profile_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one profiler mode under Nsight Systems.")
    parser.add_argument("--config", required=True, help="Path to a JSON StepProfileConfig payload.")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "forward_backward_optimizer"],
        default="forward_backward_optimizer",
        help="Profiler mode to execute inside the Nsight capture.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        payload = json.load(f)

    payload["nsight_enabled"] = False
    config = StepProfileConfig(**payload)
    result = profile_mode(config, args.mode)  # type: ignore[arg-type]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
