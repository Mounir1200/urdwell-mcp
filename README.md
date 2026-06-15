# ContextMemory

Long-term memory MCP server for LLM agents. It combines an append-only raw
archive with structured, bi-temporal memories that support invalidation and
confidence scores.

## Architecture

```
context_memory.py  MCP tools exposed to the LLM
pipeline.py        consolidation: similarity -> ADD/IGNORE/EXPIRE
embeddings.py      multilingual text embeddings
storage.py         archive, memories, and embedding persistence
models.py          bi-temporal Memory model
```

## Install

```bash
uv sync
```

If PyTorch installation fails on Python 3.14, pin Python 3.13 with
`uv python pin 3.13`, then run `uv sync` again.

The embedding model is downloaded and cached on first use.

## Test

```bash
uv run python -m unittest -v
uv run mcp dev context_memory.py
```

The unittest suite includes real MCP end-to-end tests over stdio. It starts the
server with an isolated temporary data directory and a deterministic local
embedding backend, so tests never modify production memories or download a
model. The second command opens MCP Inspector for manual testing.

LongMemEval retrieval benchmark:

```bash
uv run python benchmarks/longmemeval/run_retrieval.py \
  --backend transformer \
  --granularity turn \
  --limit 100
```

See `benchmarks/longmemeval/README.md` for the full protocol and commands.

The same guide includes the end-to-end evaluation that replays conversations,
extracts and consolidates memories, generates answers, and applies the official
LongMemEval judge rubric.

Free local Ollama smoke test:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\benchmarks\longmemeval\run_local.ps1 `
  -Dataset s `
  -RunName local-smoke-3 `
  -Limit 3 `
  -VerboseOutput
```

Optional environment variables:

- `CONTEXT_MEMORY_DATA_DIR`: override the persistence directory.
- `CONTEXT_MEMORY_EMBEDDING_BACKEND`: `transformer` by default, or `hashing`
  for deterministic tests and offline diagnostics.

## Claude Desktop

Add the server to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ContextMemory": {
      "command": "uv",
      "args": ["run", "--directory", "C:\\YOUR\\TO\\PATH\\ContextMemory", "context_memory.py"]
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `save_memory` | Stores a memory and requests arbitration when needed |
| `archive_exchange` | Appends an exact exchange to the raw archive |
| `search_memory` | Searches active memories semantically |
| `list_memories` | Lists memories by type |
| `check_conflicts` | Detects potential contradictions without writing |
| `read_archive` | Reads recent raw exchanges |

Memory types: `fact`, `preference`, `decision`, and `temporary_state`.

## Roadmap

- [ ] Confidence decay for `temporary_state` memories
- [x] Conflict resolution by the calling LLM
- [x] LongMemEval retrieval integration
- [x] LongMemEval end-to-end evaluation harness
- [ ] Establish and improve the full LongMemEval-S score
