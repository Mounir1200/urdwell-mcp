from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from benchmarks.longmemeval.llm_client import ChatClient


class FakeChatHandler(BaseHTTPRequestHandler):
    requests = []
    reject_modern_parameters = False

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(payload)

        if (
            self.__class__.reject_modern_parameters
            and "max_completion_tokens" in payload
        ):
            body = json.dumps({"error": {"message": "unknown parameter"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class ChatClientTests(unittest.TestCase):
    def setUp(self):
        FakeChatHandler.requests = []
        FakeChatHandler.reject_modern_parameters = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChatHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_modern_chat_completion_payload(self):
        client = ChatClient(
            self.base_url,
            "secret",
            "test-model",
            reasoning_effort="low",
        )

        response = client.complete("hello", max_tokens=50, json_mode=True)

        self.assertEqual(response.content, "ok")
        self.assertEqual(response.prompt_tokens, 4)
        payload = FakeChatHandler.requests[0]
        self.assertEqual(payload["max_completion_tokens"], 50)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_legacy_compatibility_fallback(self):
        FakeChatHandler.reject_modern_parameters = True
        client = ChatClient(
            self.base_url,
            "",
            "local-model",
            max_retries=2,
            reasoning_effort="low",
        )

        response = client.complete("hello", max_tokens=50, json_mode=True)

        self.assertEqual(response.content, "ok")
        self.assertEqual(len(FakeChatHandler.requests), 2)
        fallback = FakeChatHandler.requests[1]
        self.assertEqual(fallback["max_tokens"], 50)
        self.assertNotIn("max_completion_tokens", fallback)
        self.assertEqual(fallback["reasoning_effort"], "low")
        self.assertEqual(fallback["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
