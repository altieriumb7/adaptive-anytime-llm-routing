import os, time, random
from dataclasses import dataclass
from typing import Optional, List

from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APITimeoutError, APIStatusError


@dataclass
class OpenAITeacher:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    store: bool = False

    def __post_init__(self):
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        # aumenta i retry automatici del client (default è basso)
        self.client = OpenAI(api_key=key, max_retries=8)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        max_attempts: int = 12,
    ) -> str:
        last_err = None

        for attempt in range(max_attempts):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    store=self.store,
                )

                text = getattr(resp, "output_text", None)
                if not text:
                    # fallback robusto
                    text = resp.output[0].content[0].text

                if stop_strings:
                    cut = len(text)
                    for s in stop_strings:
                        idx = text.find(s)
                        if idx != -1:
                            cut = min(cut, idx)
                    text = text[:cut]

                return text.strip()

            except RateLimitError as e:
                last_err = e
                # exponential backoff + jitter
                sleep_s = min(60.0, (2 ** attempt)) * (0.5 + random.random())
                time.sleep(sleep_s)

            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                last_err = e
                # backoff più corto per errori transient
                sleep_s = min(30.0, (2 ** attempt)) * (0.5 + random.random())
                time.sleep(sleep_s)

        raise last_err
