# openai_adapter.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class OpenAIAdapter:
    """
    Tiny wrapper around OpenAI Responses API.
    Keep this file as the only OpenAI-specific part of your system.
    """
    api_key_env: str = "OPENAI_API_KEY"

    def __post_init__(self):
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set environment variable {self.api_key_env}."
            )
        self.client = OpenAI(api_key=api_key)

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
        """
        Returns plain text output.
        Uses Responses API: client.responses.create(...).output_text
        """
        # Basic sanity checks (keep it simple & safe)
        temperature = float(max(0.0, min(2.0, temperature)))
        max_output_tokens = int(max(1, min(1000, max_output_tokens)))  # server-side cap

        resp = self.client.responses.create(
            model=model,
            input=prompt,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            store=store,  # prefer false for teaching/privacy
        )
        return resp.output_text

    def list_models(self) -> list[dict]:
        models = self.client.models.list()
        return [
            {
                "id": model.id,
                "object": getattr(model, "object", "model"),
                "created": getattr(model, "created", None),
                "owned_by": getattr(model, "owned_by", None),
            }
            for model in models.data
        ]

    def list_model_ids(self) -> list[str]:
        return [model["id"] for model in self.list_models() if model.get("id")]
