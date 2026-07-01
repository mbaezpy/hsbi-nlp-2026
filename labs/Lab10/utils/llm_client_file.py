# llm_client_file.py
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional


class LLMClient:
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "kiconnect": "Mistral Small 4",
    }

    def __init__(
        self,
        base_dir: str,
        timeout_s: float = 60.0,
        poll_interval: float = 0.2,
        provider: str = "openai",
        model: Optional[str] = None,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.req_dir = self.base_dir / "requests"
        self.resp_dir = self.base_dir / "responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.poll_interval = poll_interval
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider, self.DEFAULT_MODELS["openai"])

    def prompt(
        self,
        prompt_text: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 200,
        instructions: Optional[str] = None,
    ) -> str:
        provider = provider or self.provider
        model = model or self.model or self.DEFAULT_MODELS.get(provider, self.DEFAULT_MODELS["openai"])
        data = self._request(
            action="generate",
            provider=provider,
            payload={
                "prompt": prompt_text,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "instructions": instructions,
            },
        )
        return data["response"]

    def list_models(self, *, provider: Optional[str] = None) -> list[dict]:
        provider = provider or self.provider
        data = self._request(action="list_models", provider=provider, payload={})
        return data["models"]

    def list_model_ids(self, *, provider: Optional[str] = None) -> list[str]:
        return [model["id"] for model in self.list_models(provider=provider) if model.get("id")]

    def _request(self, *, action: str, provider: str, payload: dict) -> dict:
        req_id = str(uuid.uuid4())
        req_path = self.req_dir / f"{req_id}.json"
        tmp_req_path = self.req_dir / f"{req_id}.json.tmp"
        resp_path = self.resp_dir / f"{req_id}.json"

        payload = {
            "id": req_id,
            "action": action,
            "provider": provider,
            **payload,
        }

        with tmp_req_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp_req_path.replace(req_path)

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if resp_path.exists():
                try:
                    with resp_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    time.sleep(self.poll_interval)
                    continue
                if not data.get("ok", False):
                    raise RuntimeError(f"LLM server error: {data.get('error')}")
                return data
            time.sleep(self.poll_interval)

        raise TimeoutError(f"No response after {self.timeout_s}s (req_id={req_id}).")
