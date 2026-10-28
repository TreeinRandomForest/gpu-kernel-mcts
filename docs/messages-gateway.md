# EnMaaS gateway for GPU search

**Verified GLM-5.3 gateway (2026-10-28):** A request to the company Messages
endpoint configured in `~/.bashrc` returned HTTP 200, model
`rits/zai-org/glm-5-3`, and token usage.
The repository's `MessagesClient` then returned `OK` for the same smoke prompt.
The older dogfood gateway host rejected this key with HTTP 401; that failure
did not establish that the key itself was invalid.

`search_cli` supports `--generator messages` for CUDA C++ MCTS and sequential
ablations. It uses the same source prompts, verification pipeline, profiling,
trace schema, and `B_gen` accounting as the OpenAI generator. MCTS sends a
fresh single-turn request for every generation or repair; sequential search
replays the configured number of visible text turns. This option does not
change CuTe or search selection rules.

Set the company gateway URL and key in the shell that launches the controller:

```bash
export ENMAAS_BASE_URL='https://YOUR_ENMAAS_HOST/v1'
# Set ENMAAS_KEY using your approved secret-management method.
# If .bashrc defines it without export: source ~/.bashrc; export ENMAAS_KEY
```

The default key variable is `ENMAAS_KEY`; the CLI never takes the key value
as an argument. The CLI also accepts `ENMAAS_BASE_URL` ending in either `/v1`
or `/v1/messages`; it appends `/messages` only when needed. `ENMAAS_URL` takes
precedence if both variables are set, and `--messages-url` overrides either.
The URL can therefore be passed explicitly when desired. The
client requires HTTPS, sends `x-api-key`, `anthropic-version: 2023-06-01`,
and JSON, and parses Anthropic-style text blocks and usage. It does not send
the OpenAI-only `reasoning_effort` parameter. The trace records model ID,
provider style, API version, an endpoint **hash**, exact prompts, output,
and available usage counts; it does not record the key or endpoint URL.

The GLM-5.3 Triton smoke exposed a gateway behavior: with the default Messages
request, the model consumed 16,384 output tokens and returned no text for the
actual first-generation prompt. The gateway accepted
`thinking: {"type": "disabled"}`; the same prompt then returned source text
in about five seconds. Use `--messages-thinking disabled` for this GLM search
configuration. The default remains `default` (no `thinking` field) for other
models or experiments. Both the run metadata and each generation record the
selected mode. This controls response generation, not input context length.

To run MCTS, use the existing worker arguments and add:

```bash
--generator messages \
--model rits/zai-org/glm-5-3 \
--messages-thinking disabled \
--strategies configs/strategies.yaml \
--generation-budget 50
```

For a sequential freeform arm, add `--algorithm sequential` and the desired
context/history flags; `--session-history-turns 2` is the earlier bounded
history policy. For an otherwise matched explicit-strategy arm, add
`--sequential-strategy-menu --strategies configs/strategies.yaml`. Keep the
root, H100 image, NCU bundle, compiler/dataflow views, history depth,
generation budget, and seed fixed across arms. The endpoint and GLM model
can be recorded in a private experiment record if needed; the trace stores
only the endpoint hash.

The API-only smoke and a Triton H100 `B_gen=3` run have passed; see
`docs/experiments/triton-h100-glm53-smoke-2026-10-28.md`. HTTP failures are
reported without response bodies or credentials. The model API and GPU worker
each require their own network access and credentials.

The runner also supports a configurable OpenAI-compatible Responses base URL:
`--generator openai --openai-base-url BASE_URL --openai-api-key-env ENMAAS_KEY`.
On 2026-10-28, a direct GLM Responses request to the configured company
gateway succeeded with
`reasoning.effort="high"`, and an ephemeral Codex profile using a temporary
base-URL and high-effort override completed with `OK`. The saved Codex
provider definition still targets the dogfood host; update its `base_url` to
the current company gateway for persistent Codex use. This confirms
request acceptance, not a quality difference between effort levels. The
Messages route remains the first search smoke path because its repository
adapter was also verified end to end.
