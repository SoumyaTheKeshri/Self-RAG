"""
Single wrapper around the LLM API. Every other file calls `call_llm()`
instead of touching the OpenAI client directly — so swapping providers
later (Anthropic, a local model via Ollama, etc.) only means editing
this one file.
"""

from openai import OpenAI
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1")


def call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
    """
    Send a single-turn prompt to the chat model and return the text response.
    temperature=0.0 -> deterministic (used for critique/judging steps)
    temperature>0.0 -> varied (used for generating multiple candidates)
    """
    try:
        response = _client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fail loudly but don't crash the whole pipeline on one bad call
        print(f"[llm.py] LLM call failed: {e}")
        return ""
