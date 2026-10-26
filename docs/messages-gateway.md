# Anthropic-compatible Messages gateway for CUDA search

`search_cli` supports `--generator messages` for CUDA C++ MCTS and sequential
ablations. It uses the same source prompts, verification pipeline, profiling,
trace schema, and `B_gen` accounting as the OpenAI generator. MCTS sends a
fresh single-turn request for every generation or repair; sequential search
replays the configured number of visible text turns. This option does not
change CuTe or search selection rules.

Set the company gateway URL and key in the shell that launches the controller:

```bash
export ENMAAS_URL='https://YOUR-GATEWAY/v1/messages'
# Set ENMAAS_KEY using your approved secret-management method.
# If .bashrc defines it without export: source ~/.bashrc; export ENMAAS_KEY
```

The default key variable is `ENMAAS_KEY`; the CLI never takes the key value
as an argument. The URL can instead be passed with `--messages-url`. The
client requires HTTPS, sends `x-api-key`, `anthropic-version: 2023-06-01`,
and JSON, and parses Anthropic-style text blocks and usage. It does not send
the OpenAI-only `reasoning_effort` parameter. The trace records model ID,
provider style, API version, an endpoint **hash**, exact prompts, output,
and available usage counts; it does not record the key or endpoint URL.

To run MCTS, use the existing worker arguments and add:

```bash
--generator messages \
--model rits/zai-org/glm-5-3 \
--strategies configs/strategies.yaml \
--generation-budget 50
```

For a sequential freeform arm, add `--algorithm sequential` and the desired
context/history flags; `--session-history-turns 2` is the earlier bounded
history policy. For an otherwise matched explicit-strategy arm, add
`--sequential-strategy-menu --strategies configs/strategies.yaml`. Keep the
root, H100 image, NCU bundle, compiler/dataflow views, history depth,
generation budget, and seed fixed across arms. The endpoint and GLM model
should be recorded in the experiment report; the trace stores only the
endpoint hash.

Start with a single request against the gateway and a small `B_gen` GPU
smoke before a costly ablation. HTTP failures are reported without response
bodies or credentials. The model API and GPU worker each require their own
network access and credentials. The gateway integration has GPU-free mocked
tests. On 2026-10-26, a live request through the adapter and a request with
the user-provided curl shape both returned HTTP 401. The key was nonempty
with no leading/trailing whitespace or newline, but was not exported by
`.bashrc`. Exporting it made the request reach the gateway; authentication
still failed. No H100 validation has been run for this model.
