# RunPod profiler worker

This worker runs `language_machine.profile_step.run_step_profile` on RunPod Serverless.

Build and push the worker image:

```bash
cd infrastructure
./runpod_profiler/build_and_push.sh docker.io/<docker-user>/language-machine-profiler:latest
```

Create a RunPod Serverless endpoint from that image. The dashboard backend calls RunPod synchronously through `/runsync`.

Start the local dashboard backend with:

```bash
RUNPOD_ENDPOINT_ID=<endpoint-id> \
RUNPOD_API_KEY=<api-key> \
RUNPOD_TIMEOUT=900 \
./start.sh
```

The Profiler page can then run with `device` set to `cuda` or `cuda:0`.

## Nsight Systems traces

The profiler request supports optional Nsight Systems capture:

- `nsight_enabled`: set to `true` to run a separate `nsys profile` capture after the normal timing pass.
- `nsight_mode`: one of `forward`, `forward_backward`, or `forward_backward_optimizer`.
- `nsight_output_dir`: directory where `.nsys-rep` artifacts are written inside the worker.

The normal timing results are collected before Nsight runs, so trace overhead does not affect `tokens_per_sec`.
The response includes Nsight metadata and generated file paths under `nsight`.

For RunPod Serverless, set `nsight_output_dir` to a persistent volume path if you need to retrieve the `.nsys-rep` file after the request completes. Otherwise the response can show that capture succeeded, but files written to an ephemeral container may not be available later.
