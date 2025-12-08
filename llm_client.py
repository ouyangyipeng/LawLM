import logging
import time
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        timeout: int = 60,
        max_retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        # Hardcoded defaults per user request; still allow overrides via parameters.
        self.api_key = api_key or "sk-64eefce97b664c8d8d45ed76a012a738"
        self.base_url = base_url or "https://api.deepseek.com"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def send_chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        extra_messages: Optional[List[ChatCompletionMessageParam]] = None,
    ) -> str:
        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if extra_messages:
            messages.extend(extra_messages)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=messages,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # Broad catch to ensure retry.
                wait_time = self.backoff ** attempt
                self.logger.warning(
                    "LLM request failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait_time,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(wait_time)
        return ""
