#!/usr/bin/env bash

set -euo pipefail

: "${NEBIUS_PROJECT_ID:?export NEBIUS_PROJECT_ID}"
: "${NEBIUS_SUBNET_ID:?export NEBIUS_SUBNET_ID}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY for the mixed ablations}"

image="${IMAGE:-docker.io/saarora/gpu-kernel-mcts:cutedsl-ablation-v82}"
model="${MODEL:-gpt-5.6-terra}"
mutation_budget="${MUTATION_BUDGET:-20}"
generation_budget="${GENERATION_BUDGET:-20}"
tuning_budget="${TUNING_BUDGET:-5}"
seed="${SEED:-0}"
output_dir="${OUTPUT_DIR:-ablation-results}"
ssh_key="${NEBIUS_SSH_KEY:-$HOME/.ssh/nebius}"
dry_run="${DRY_RUN:-0}"

mkdir -p "$output_dir"

common=(
  --provider nebius
  --image "$image"
  --backend cute_dsl
  --cute-root-kind independent
  --mutation-budget "$mutation_budget"
  --c-puct 1.5
  --k-max 4
  --max-depth 10
  --seed "$seed"
  --profile-metric-set lightweight_v1
  --nebius-project-id "$NEBIUS_PROJECT_ID"
  --nebius-subnet-id "$NEBIUS_SUBNET_ID"
  --nebius-ssh-private-key "$ssh_key"
  --nebius-ssh-public-key "${ssh_key}.pub"
  --timeout 1200
  --confirm-create-and-terminate
)

run_ablation() {
  local name="$1"
  shift
  local trace="$output_dir/${name}.sqlite"
  local best="$output_dir/${name}-best.py"
  local log="$output_dir/${name}.log"

  if [[ -e "$trace" || -e "$best" || -e "$log" ]]; then
    echo "Refusing to overwrite an existing artifact for $name" >&2
    return 1
  fi

  local command=(
    .venv/bin/python -m kernel_mcts.search_cli
    "${common[@]}"
    --trace "$trace"
    --best-output "$best"
    "$@"
  )
  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi
  "${command[@]}" 2>&1 | tee "$log"
}

run_ablation typed-only \
  --generator cute-mutation \
  --generation-budget 0

run_ablation typed-tuned \
  --generator cute-mutation \
  --generation-budget 0 \
  --autotune \
  --tuning-budget "$tuning_budget" \
  --tuning-method grid \
  --tuned-best-output "$output_dir/typed-tuned-final.py"

run_ablation typed-llm \
  --generator cute-mixed \
  --generation-budget "$generation_budget" \
  --model "$model" \
  --reasoning-effort medium

run_ablation typed-llm-tuned \
  --generator cute-mixed \
  --generation-budget "$generation_budget" \
  --model "$model" \
  --reasoning-effort medium \
  --autotune \
  --tuning-budget "$tuning_budget" \
  --tuning-method grid \
  --tuned-best-output "$output_dir/typed-llm-tuned-final.py"
