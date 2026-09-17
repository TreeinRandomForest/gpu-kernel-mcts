#!/bin/bash

set -euo pipefail

provider="nebius"
tune_budget=18
method="grid"
kernel="nebius-sol-best-bgen50-cpuct12-variantdiagnostic_v2-deltas-off.cu"
variant="wmma-grid-btune${tune_budget}"
log="log-nebius-sol-tuning-${variant}"

.venv/bin/python -m kernel_mcts.autotune_cli \
  --input "$kernel" \
  --output "nebius-sol-tuned-${variant}.cu" \
  --trace "nebius-sol-tuning-${variant}.sqlite" \
  --image docker.io/saarora/gpu-kernel-mcts:h100-worker-v11 \
  --tuning-budget "$tune_budget" \
  --tuning-method "$method" \
  --seed 0 \
  --nebius-project-id project-e00aymjmpr00grzcky46ba \
  --nebius-subnet-id vpcsubnet-e00a30bhyztjvq7yaz \
  --nebius-username sanjay \
  --nebius-ssh-private-key "$HOME/.ssh/${provider}" \
  --nebius-ssh-public-key "$HOME/.ssh/${provider}.pub" \
  --timeout 1200 \
  --confirm-create-and-terminate 2>&1 | tee "$log"
