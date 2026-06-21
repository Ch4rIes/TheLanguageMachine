#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <image-tag>"
  echo "example: $0 docker.io/your-user/language-machine-profiler:latest"
  exit 2
fi

IMAGE_TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_CONTEXT="$(mktemp -d)"

cleanup() {
  rm -rf "$BUILD_CONTEXT"
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the RunPod worker image"
  exit 127
fi

mkdir -p "$BUILD_CONTEXT/core" "$BUILD_CONTEXT/infrastructure"
cp -R "$REPO_ROOT/core/language_machine" "$BUILD_CONTEXT/core/language_machine"
cp -R "$REPO_ROOT/infrastructure/runpod_profiler" "$BUILD_CONTEXT/infrastructure/runpod_profiler"

docker build --platform linux/amd64 \
  -f "$BUILD_CONTEXT/infrastructure/runpod_profiler/Dockerfile" \
  -t "$IMAGE_TAG" "$BUILD_CONTEXT"

docker push "$IMAGE_TAG"
