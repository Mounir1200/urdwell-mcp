# Security Policy

## Supported versions

ContextMemory is pre-1.0. Security fixes are applied to the latest release only.

## Threat model

ContextMemory runs as a **local process** launched by an MCP client (for example
Claude Desktop) over **stdio**. It opens **no network port** and exposes no
remote attack surface in its default configuration. The data store is a set of
local Parquet files owned by the user who runs the server.

As a result, the security considerations below are about **data integrity** and
**context safety**, not remote code execution.

## What you should know before using it

### Memory poisoning / stored prompt injection
Anything written through `save_memory` or `archive_exchange` is later read back
into a language model through `search_memory` or `read_archive`. If an agent
ingests hostile content (for example a web page or document that contains
"ignore your instructions and ..."), that text can be stored and **resurface as
an apparent instruction** in a later session.

This is inherent to any long-term memory tool. The consuming agent must treat
stored memory content as **data, never as instructions**, and you should only
grant memory-writing tools to agents whose input sources you trust.

### Embedding model is a supply-chain trust boundary
With the default `fastembed` backend, the ONNX embedding model is downloaded
from Hugging Face on first use. Loading remote model data remains a supply-chain
trust boundary. ContextMemory does not install or execute PyTorch.

For fully offline, dependency-free operation, use the deterministic backend:

```bash
CONTEXT_MEMORY_EMBEDDING_BACKEND=hashing
```

### Local data is stored unencrypted
Memories, embeddings, and the raw archive are written as unencrypted Parquet under the
data directory (`CONTEXT_MEMORY_DATA_DIR`, default `data/`). Anyone with read
access to that directory can read every stored memory. Place the data directory
on storage with appropriate filesystem permissions, and do not commit it (it is
ignored by Git by default).

## Hardening already in place

- Writes are **atomic** (temp file + `os.replace`) and serialized with a
  process-wide lock, so a crash or concurrent tool calls cannot corrupt or lose
  the store.
- Tool inputs are bounded (content length, result count, archive read size) and
  `confidence` is clamped to `[0, 1]`.
- Persistence uses PyArrow Parquet — no `pickle`, `eval`, `exec`, or shell calls.
- Tool arguments are never used to build filesystem paths, so there is no path
  traversal from model-supplied input.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to **mdabire05@gmail.com**
rather than opening a public issue. Include a description, affected version, and
reproduction steps. You can expect an initial acknowledgement within a few days.
