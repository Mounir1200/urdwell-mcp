import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from urdwell import integrations as ig


class McpServersWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "mcp.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_configure_adds_entry_and_preserves_unrelated_content(self):
        self.path.write_text(
            json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"})
        )

        ig._configure_mcpservers(self.path)

        data = json.loads(self.path.read_text())
        self.assertEqual(
            data["mcpServers"]["urdwell"],
            {"command": "urdwell", "args": ["serve"]},
        )
        self.assertIn("other", data["mcpServers"])
        self.assertEqual(data["theme"], "dark")

    def test_configure_is_idempotent(self):
        ig._configure_mcpservers(self.path)
        ig._configure_mcpservers(self.path)

        data = json.loads(self.path.read_text())
        self.assertEqual(list(data["mcpServers"]), ["urdwell"])

    def test_configure_replaces_legacy_server_entry(self):
        self.path.write_text(
            json.dumps({"mcpServers": {"contextmemory": {"command": "old"}}})
        )

        ig._configure_mcpservers(self.path)

        servers = json.loads(self.path.read_text())["mcpServers"]
        self.assertNotIn("contextmemory", servers)
        self.assertEqual(
            servers["urdwell"],
            {"command": "urdwell", "args": ["serve"]},
        )

    def test_unconfigure_removes_only_our_entry(self):
        self.path.write_text(
            json.dumps(
                {"mcpServers": {"urdwell": {}, "contextmemory": {}, "other": {}}}
            )
        )

        self.assertTrue(ig._unconfigure_mcpservers(self.path))

        servers = json.loads(self.path.read_text())["mcpServers"]
        self.assertNotIn("urdwell", servers)
        self.assertNotIn("contextmemory", servers)
        self.assertIn("other", servers)

    def test_unconfigure_returns_false_when_absent(self):
        self.path.write_text(json.dumps({"mcpServers": {"other": {}}}))
        self.assertFalse(ig._unconfigure_mcpservers(self.path))

    def test_invalid_json_is_skipped_rather_than_corrupted(self):
        self.path.write_text("{ not json // with a comment }")
        with self.assertRaises(ig._SkipAgent):
            ig._configure_mcpservers(self.path)


class OpencodeWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "opencode.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_configure_writes_local_server_and_schema(self):
        ig._configure_opencode(self.path)

        data = json.loads(self.path.read_text())
        self.assertEqual(data["$schema"], "https://opencode.ai/config.json")
        self.assertEqual(
            data["mcp"]["urdwell"],
            {"type": "local", "command": ["urdwell", "serve"], "enabled": True},
        )

    def test_unconfigure_removes_entry(self):
        ig._configure_opencode(self.path)
        self.assertTrue(ig._unconfigure_opencode(self.path))
        self.assertNotIn(
            "urdwell", json.loads(self.path.read_text()).get("mcp", {})
        )


class CodexTomlWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "config.toml"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_configure_preserves_existing_sections_and_is_idempotent(self):
        self.path.write_text("[other]\nkey = 1\n")

        ig._configure_codex(self.path)
        ig._configure_codex(self.path)

        text = self.path.read_text()
        self.assertEqual(text.count("[mcp_servers.urdwell]"), 1)
        parsed = tomllib.loads(text)
        self.assertEqual(
            parsed["mcp_servers"]["urdwell"],
            {"command": "urdwell", "args": ["serve"]},
        )
        self.assertEqual(parsed["other"]["key"], 1)

    def test_configure_replaces_legacy_codex_section(self):
        self.path.write_text(
            '[mcp_servers.contextmemory]\ncommand = "contextmemory"\nargs = ["serve"]\n'
            "\n[other]\nkey = 3\n"
        )

        ig._configure_codex(self.path)

        parsed = tomllib.loads(self.path.read_text())
        self.assertNotIn("contextmemory", parsed.get("mcp_servers", {}))
        self.assertEqual(
            parsed["mcp_servers"]["urdwell"],
            {"command": "urdwell", "args": ["serve"]},
        )
        self.assertEqual(parsed["other"]["key"], 3)

    def test_unconfigure_removes_our_section_and_keeps_the_next(self):
        ig._configure_codex(self.path)
        self.path.write_text(self.path.read_text() + "\n[other]\nkey = 2\n")

        self.assertTrue(ig._unconfigure_codex(self.path))

        parsed = tomllib.loads(self.path.read_text())
        self.assertNotIn("mcp_servers", parsed)
        self.assertEqual(parsed["other"]["key"], 2)

    def test_unconfigure_returns_false_when_absent(self):
        self.path.write_text("[other]\nkey = 1\n")
        self.assertFalse(ig._unconfigure_codex(self.path))


class RegistryTests(unittest.TestCase):
    def test_agent_keys_are_unique(self):
        keys = [agent.key for agent in ig.REGISTRY]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_agent_is_callable(self):
        for agent in ig.REGISTRY:
            self.assertTrue(callable(agent.detect))
            self.assertTrue(callable(agent.configure))
            self.assertTrue(callable(agent.unconfigure))


if __name__ == "__main__":
    unittest.main()
