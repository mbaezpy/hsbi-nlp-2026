from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


def _load_env_file(env_path: Optional[Path] = None) -> None:
    path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class KiConnectAdapter:
    """
    Wrapper around the KI:connect API.
    Keep this file as the only KI:connect-specific part of your system.
    """

    api_key_env: str = "KICONNECT_KEY"
    base_url_env: str = "KICONNECT_BASE_URL"
    default_base_url: str = "https://chat.kiconnect.nrw/app/api/v1"
    timeout_s: float = 120.0

    def __post_init__(self):
        _load_env_file()

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set environment variable {self.api_key_env}."
            )

        self.api_key = api_key
        self.base_url = os.getenv(self.base_url_env, self.default_base_url).rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        body = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                details = json.loads(raw)
            except json.JSONDecodeError:
                details = raw
            raise RuntimeError(f"KiConnect API error {exc.code}: {details}") from exc

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 200,
        instructions: Optional[str] = None,
        store: bool = False,
    ) -> str:
        temperature = float(max(0.0, min(2.0, temperature)))
        max_output_tokens = int(max(1, min(4000, max_output_tokens)))

        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
        }

        # Some models on KI:connect reject custom temperatures.
        if temperature != 1.0:
            payload["temperature"] = temperature

        try:
            data = self._request("POST", "/chat/completions", payload=payload)
        except RuntimeError as exc:
            msg = str(exc)
            if "temperature" in msg and "Unsupported" in msg and "temperature" in payload:
                payload.pop("temperature", None)
                data = self._request("POST", "/chat/completions", payload=payload)
            else:
                raise

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"KiConnect returned no choices: {data}")

        message = choices[0].get("message") or {}
        text = self._extract_text(message.get("content"))
        if text:
            return text

        finish_reason = choices[0].get("finish_reason")
        raise RuntimeError(
            "KiConnect returned an empty response"
            + (f" (finish_reason={finish_reason})" if finish_reason else "")
        )

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/models")
        return data.get("data", [])

    def list_model_ids(self) -> list[str]:
        return [model.get("id", "") for model in self.list_models() if model.get("id")]
