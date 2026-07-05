"""
BASICS #1 (Day 1): Your first RAW API call (no framework!)
===========================================================
Checkpoint: you see a model response printed in your terminal.

Before LangChain, before CrewAI -- this is what talking to an LLM actually
looks like: an HTTPS POST request with your API key and some JSON. Every
framework you'll learn this week is just a nicer wrapper around THIS.

Setup (do once):
  1. Copy basics/.env.example to basics/.env
  2. Put your free API key in it (Gemini, Groq, or OpenRouter)
  3. pip install -r ../project/requirements.txt

Run with:
    python 01_first_api_call.py

Test it (from the repo root):
    python check.py 01
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()  # <-- classic gotcha: forget this line and os.getenv finds nothing!

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# TODO: Student Code Here
# Write ANY prompt you like -- ask the model something fun. One string.
# Example: "Explain what an AI agent is, in exactly 2 sentences."
# =============================================================================
PROMPT = "Explain what an AI agent is, in exactly 2 sentences, to a college student."


# -----------------------------------------------------------------------------
# Provider plumbing -- provided for you. Skim it! Notice there's no magic:
# each function is just requests.post(url, json=...) with a key.
# -----------------------------------------------------------------------------
def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    # .env stores the model as "gemini/gemini-2.5-flash" (a LiteLLM id used
    # on Day 2); the raw REST API wants just "gemini-2.5-flash".
    model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash").split("/")[-1]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    response = requests.post(
        url,
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile").split("/", 1)[-1]

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    ).split("/", 1)[-1]

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    if "TODO" in PROMPT:
        print("✗ You haven't written your prompt yet -- edit the PROMPT variable first!")
        sys.exit(1)

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    callers = {"gemini": call_gemini, "groq": call_groq, "openrouter": call_openrouter}

    if provider not in callers:
        print(f"✗ Unknown LLM_PROVIDER '{provider}' -- use gemini, groq, or openrouter.")
        sys.exit(1)

    print(f"Asking {provider}: {PROMPT!r}\n")
    answer = callers[provider](prompt=PROMPT)

    print("--- MODEL RESPONSE " + "-" * 40)
    print(answer.strip())
    print("-" * 59)
    print("\n✅ CHECKPOINT 1 PASSED: your API key works and you got a real model response.")


if __name__ == "__main__":
    main()
