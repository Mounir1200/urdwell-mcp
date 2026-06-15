"""Minimal OpenAI-compatible chat completions client for benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ChatResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 180,
        max_retries: int = 6,
        reasoning_effort: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        compatibility_stage = 0
        for attempt in range(self.max_retries):
            encoded = json.dumps(payload).encode("utf-8")
            request = Request(
                self.endpoint,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = body["choices"][0]["message"]["content"]
                if isinstance(choice, list):
                    choice = "".join(
                        item.get("text", "")
                        for item in choice
                        if isinstance(item, dict)
                    )
                usage = body.get("usage", {})
                return ChatResponse(
                    content=str(choice).strip(),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                )
            except HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")
                error.close()
                if (
                    error.code == 400
                    and compatibility_stage == 0
                    and "max_completion_tokens" in payload
                ):
                    payload["max_tokens"] = payload.pop("max_completion_tokens")
                    compatibility_stage = 1
                    continue
                if error.code == 400 and compatibility_stage == 1:
                    payload.pop("reasoning_effort", None)
                    payload.pop("response_format", None)
                    compatibility_stage = 2
                    continue
                last_error = RuntimeError(
                    f"LLM HTTP {error.code}: {error_body[:1000]}"
                )
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from error
            except (URLError, TimeoutError, KeyError, ValueError) as error:
                last_error = error

            if attempt + 1 < self.max_retries:
                time.sleep(min(2**attempt, 30))

        raise RuntimeError(
            f"LLM request failed after {self.max_retries} attempts: {last_error}"
        )
