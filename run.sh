#!/bin/bash

set -euo pipefail

provider="nebius"
gen_budget=50
model="sol"
cpuct=12
profile_set="diagnostic_v2"

variant=${profile_set}-deltas-on

log=log-${provider}-${model}-bgen${gen_budget}-cpuct${cpuct}-variant${variant}

.venv/bin/python -m kernel_mcts.search_cli \
      --provider "$provider" \
      --image docker.io/saarora/gpu-kernel-mcts:h100-worker-v11 \
      --trace "${provider}-${model}-bgen${gen_budget}-cpuct${cpuct}-variant${variant}.sqlite" \
      --include-incoming-profile-delta \
      --nebius-project-id project-e00aymjmpr00grzcky46ba \
      --nebius-subnet-id vpcsubnet-e00a30bhyztjvq7yaz \
      --nebius-username sanjay \
      --nebius-ssh-private-key "$HOME/.ssh/${provider}" \
      --nebius-ssh-public-key "$HOME/.ssh/${provider}.pub" \
      --generator openai \
      --model "gpt-5.6-${model}" \
      --reasoning-effort medium \
      --profile-metric-set "$profile_set" \
      --strategies configs/strategies.yaml \
      --generation-budget "$gen_budget" \
      --max-repairs 1 \
      --c-puct "$cpuct" \
      --k-max 4 \
      --max-depth 10 \
      --seed 0 \
      --best-output "${provider}-${model}-best-bgen${gen_budget}-cpuct${cpuct}-variant${variant}.cu" \
      --timeout 1200 \
      --confirm-create-and-terminate 2>&1 | tee "$log"


#  systemd-inhibit --what=sleep:idle --why="GPU MCTS search" --mode=block bash ./run.sh
