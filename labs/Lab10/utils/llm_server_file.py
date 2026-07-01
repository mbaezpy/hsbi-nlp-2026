# llm_server_file.py
from __future__ import annotations

import json
import time
from pathlib import Path

from .kiconnect_adapter import KiConnectAdapter
from .openai_adapter import OpenAIAdapter


class LLMServer:
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "kiconnect": "Mistral Small 4",
    }

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.req_dir = self.base_dir / "requests"
        self.resp_dir = self.base_dir / "responses"
        self.done_dir = self.base_dir / "done"
        for d in (self.req_dir, self.resp_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.adapter_classes = {
            "openai": OpenAIAdapter,
            "kiconnect": KiConnectAdapter,
        }
        self.adapters = {}

        print(f"[LLMServer] base_dir = {self.base_dir}")
        print(f"[LLMServer] watching  = {self.req_dir}")

    def _get_adapter(self, provider: str):
        try:
            adapter = self.adapters.get(provider)
            if adapter is None:
                adapter = self.adapter_classes[provider]()
                self.adapters[provider] = adapter
            return adapter
        except KeyError as exc:
            known = ", ".join(sorted(self.adapter_classes))
            raise ValueError(f"Unknown provider '{provider}'. Known providers: {known}.") from exc

    def _handle_request(self, req_path: Path) -> None:
        try:
            with req_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return None

        req_id = data["id"]
        action = data.get("action", "generate")
        provider = data.get("provider", "openai")

        llm = self._get_adapter(provider)

        try:
            if action == "generate":
                prompt = data.get("prompt", "")
                model = data.get("model") or self.DEFAULT_MODELS.get(
                    provider, self.DEFAULT_MODELS["openai"]
                )
                temperature = data.get("temperature", 0.2)
                max_output_tokens = data.get("max_output_tokens", 200)
                instructions = data.get("instructions", None)

                text = llm.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                    store=False,
                )
                out = {"id": req_id, "ok": True, "response": text}
            elif action == "list_models":
                out = {"id": req_id, "ok": True, "models": llm.list_models()}
            else:
                raise ValueError(f"Unknown action '{action}'.")
        except Exception as e:
            out = {"id": req_id, "ok": False, "error": repr(e)}

        resp_path = self.resp_dir / f"{req_id}.json"
        tmp_resp_path = self.resp_dir / f"{req_id}.json.tmp"
        with tmp_resp_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        tmp_resp_path.replace(resp_path)

        # Move processed request aside
        req_path.rename(self.done_dir / req_path.name)

    def serve_forever(self, poll_interval: float = 0.2) -> None:
        while True:
            for req_path in sorted(self.req_dir.glob("*.json")):
                self._handle_request(req_path)
            time.sleep(poll_interval)


if __name__ == "__main__":
    # Use an ABSOLUTE path in practice for fewer surprises
    server = LLMServer(base_dir="./llm_bridge")
    server.serve_forever()
