"""Small MCP client for testing the server without Inspector.

It calls tools that do not use embeddings first, then exercises save_memory
with a timeout. Progress bars are disabled because noisy subprocess output can
fill an unread pipe and deadlock a stdio server.

Run with: ``uv run python test_client.py``
"""

import asyncio
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment


async def main():
    env = get_default_environment()
    env.update({
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "PYTHONUNBUFFERED": "1",
    })
    params = StdioServerParameters(
        command="uv",
        args=["run", "context_memory.py"],
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[1/5] connection established")

            tools = await session.list_tools()
            print(f"[2/5] {len(tools.tools)} tools exposed")

            r = await session.call_tool(
                "archive_exchange",
                {"role": "user", "content": "diagnostic test"},
                read_timeout_seconds=timedelta(seconds=15),
            )
            print("[3/5] archive_exchange (no model):", r.content[0].text)

            r = await session.call_tool(
                "read_archive",
                {"last_n": 3},
                read_timeout_seconds=timedelta(seconds=15),
            )
            print("[4/5] read_archive (no model): ok,", r.content[0].text[:80])

            print("[5/5] save_memory (loads model, 3 minute timeout)...")
            r = await session.call_tool(
                "save_memory",
                {
                    "content": "Mounir is building a long-term memory MCP server",
                    "memory_type": "fact",
                },
                read_timeout_seconds=timedelta(seconds=180),
            )
            print("      response:", r.content[0].text)

            # Question-to-statement similarity can be lower than similarity
            # between two statements, so test both query styles.
            for query in [
                "What is Mounir working on?",
                "long-term memory MCP server",
            ]:
                r = await session.call_tool(
                    "search_memory", {"query": query},
                    read_timeout_seconds=timedelta(seconds=60),
                )
                if r.content:
                    print(f"search({query!r}) :", r.content[0].text[:200])
                else:
                    print(f"search({query!r}): no result above threshold 0.55")


if __name__ == "__main__":
    asyncio.run(main())
