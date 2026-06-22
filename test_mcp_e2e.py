import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.types import TextContent


REPO_ROOT = Path(__file__).parent
TOOL_TIMEOUT = timedelta(seconds=10)


class McpEndToEndTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def server_session(self):
        with tempfile.TemporaryDirectory() as data_dir:
            env = get_default_environment()
            env.update(
                {
                    "URDWELL_DATA_DIR": data_dir,
                    "URDWELL_EMBEDDING_BACKEND": "hashing",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "urdwell.server"],
                env=env,
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def call_result(
        self,
        session: ClientSession,
        name: str,
        arguments: dict,
    ):
        result = await session.call_tool(
            name,
            arguments,
            read_timeout_seconds=TOOL_TIMEOUT,
        )
        self.assertFalse(result.isError)
        return result

    async def call_text(
        self,
        session: ClientSession,
        name: str,
        arguments: dict,
    ) -> str:
        result = await self.call_result(session, name, arguments)
        if result.structuredContent is not None:
            value = result.structuredContent.get(
                "result",
                result.structuredContent,
            )
            if isinstance(value, str):
                return value
            return json.dumps(value)

        text_blocks = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        ]
        self.assertTrue(text_blocks)
        return "\n".join(text_blocks)

    async def call_json(
        self,
        session: ClientSession,
        name: str,
        arguments: dict,
    ):
        result = await self.call_result(session, name, arguments)
        if result.structuredContent is not None:
            return result.structuredContent.get(
                "result",
                result.structuredContent,
            )
        text_blocks = [
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        ]
        self.assertTrue(text_blocks)
        return json.loads("\n".join(text_blocks))

    async def test_tools_expose_the_english_contract(self):
        async with self.server_session() as session:
            response = await session.list_tools()
            tools = {tool.name: tool for tool in response.tools}

            self.assertEqual(
                set(tools),
                {
                    "archive_exchange",
                    "check_conflicts",
                    "list_memories",
                    "read_archive",
                    "save_memory",
                    "search_memory",
                },
            )
            save_properties = tools["save_memory"].inputSchema["properties"]
            self.assertIn("memory_type", save_properties)
            self.assertIn("confidence", save_properties)
            self.assertIn("target_id", save_properties)
            self.assertNotIn("confiance", save_properties)
            self.assertNotIn("cible_id", save_properties)

    async def test_archive_round_trip_preserves_exact_source(self):
        async with self.server_session() as session:
            content = "Exact source: Mounir prefers concise answers."
            archived = await self.call_text(
                session,
                "archive_exchange",
                {"role": "user", "content": content, "session": "e2e"},
            )
            entries = await self.call_json(
                session,
                "read_archive",
                {"last_n": 1},
            )

            self.assertIn("archived", archived)
            self.assertEqual(entries[0]["content"], content)
            self.assertEqual(entries[0]["session"], "e2e")

    async def test_add_duplicate_and_semantic_search(self):
        async with self.server_session() as session:
            content = "The project database uses PostgreSQL."
            added = await self.call_json(
                session,
                "save_memory",
                {"content": content, "memory_type": "fact"},
            )
            duplicate = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "  the PROJECT database uses PostgreSQL.  ",
                    "memory_type": "fact",
                },
            )
            results = await self.call_json(
                session,
                "search_memory",
                {"query": "project database"},
            )
            conflicts = await self.call_json(
                session,
                "check_conflicts",
                {"content": "The project database uses MySQL."},
            )

            self.assertEqual(added["action"], "ADD")
            self.assertEqual(duplicate["action"], "IGNORE")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], added["new_memory_id"])
            self.assertEqual(conflicts[0]["id"], added["new_memory_id"])

    async def test_compatible_memories_can_coexist_after_arbitration(self):
        async with self.server_session() as session:
            await self.call_json(
                session,
                "save_memory",
                {
                    "content": "Mounir likes coffee.",
                    "memory_type": "preference",
                },
            )
            pending = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "Mounir likes tea.",
                    "memory_type": "preference",
                },
            )
            added = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "Mounir likes tea.",
                    "memory_type": "preference",
                    "decision": "ADD",
                },
            )
            memories = await self.call_json(
                session,
                "list_memories",
                {},
            )

            self.assertEqual(pending["action"], "ARBITRATION_REQUIRED")
            self.assertEqual(added["action"], "ADD")
            self.assertEqual(len(memories), 2)

    async def test_knowledge_update_expires_old_fact_and_preserves_history(self):
        async with self.server_session() as session:
            old = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "The project database uses PostgreSQL.",
                    "memory_type": "decision",
                },
            )
            pending = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "The project database uses SQLite.",
                    "memory_type": "decision",
                },
            )
            replacement = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "The project database uses SQLite.",
                    "memory_type": "decision",
                    "decision": "EXPIRE",
                    "target_id": old["new_memory_id"],
                },
            )
            active = await self.call_json(
                session,
                "list_memories",
                {},
            )
            history = await self.call_json(
                session,
                "list_memories",
                {"include_expired": True},
            )

            self.assertEqual(pending["action"], "ARBITRATION_REQUIRED")
            self.assertEqual(replacement["action"], "EXPIRE")
            self.assertEqual(len(active), 1)
            self.assertIn("SQLite", active[0]["content"])
            self.assertEqual(len(history), 2)

            old_memory = next(
                memory
                for memory in history
                if memory["id"] == old["new_memory_id"]
            )
            new_memory = next(
                memory
                for memory in history
                if memory["id"] == replacement["new_memory_id"]
            )
            self.assertIsNotNone(old_memory["valid_until"])
            self.assertEqual(new_memory["supersedes"], old_memory["id"])

    async def test_unknown_information_returns_no_search_results(self):
        async with self.server_session() as session:
            results = await self.call_json(
                session,
                "search_memory",
                {"query": "What is the user's favorite musical instrument?"},
            )

            self.assertEqual(results, [])

    async def test_invalid_memory_type_is_rejected_without_writing(self):
        async with self.server_session() as session:
            report = await self.call_json(
                session,
                "save_memory",
                {
                    "content": "This must not be stored.",
                    "memory_type": "unknown",
                },
            )
            memories = await self.call_json(
                session,
                "list_memories",
                {},
            )

            self.assertIn("error", report)
            self.assertEqual(memories, [])


if __name__ == "__main__":
    unittest.main()
