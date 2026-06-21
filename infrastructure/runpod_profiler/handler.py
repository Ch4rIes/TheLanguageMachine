import traceback

import runpod

from language_machine.profile_step import StepProfileConfig, run_step_profile


def handler(job):
    try:
        payload = job.get("input") or {}
        return run_step_profile(StepProfileConfig(**payload))
    except Exception as exc:
        return {"error": f"{exc}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
