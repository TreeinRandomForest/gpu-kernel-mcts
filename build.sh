  IMAGE=docker.io/saarora/gpu-kernel-mcts:cutedsl-diagnostic-v1
  GIT_COMMIT=$(git rev-parse HEAD)
  DIRTY_TREE=$(test -n "$(git status --porcelain)" && echo 1 || echo 0)

  docker build \
    -f Dockerfile.cutedsl \
    --build-arg KERNEL_MCTS_GIT_COMMIT="$GIT_COMMIT" \
    --build-arg KERNEL_MCTS_DIRTY_TREE="$DIRTY_TREE" \
    -t "$IMAGE" \
    .

  podman push "$IMAGE"
