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
