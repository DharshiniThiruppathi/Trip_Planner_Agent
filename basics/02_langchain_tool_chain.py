"""
BASICS #2 (Day 1): A LangChain chain + a tool-using mini agent
===============================================================
Checkpoint: your script correctly answers a "what's happening in ____ this
week" style question using LIVE search results -- something the model could
never know from its training data alone.

The assembly line analogy:
    prompt template  ->  LLM  ->  output parser
    (fill blanks)        (think)   (clean up)
LangChain's `|` operator literally pipes these together, like a factory line.

Part A builds a plain chain (no tool) and asks about CURRENT events -- watch
it fail or hedge ("I don't have real-time information...").
Part B runs a free DuckDuckGo web search FIRST, stuffs the results into the
prompt, and asks again -- now the model can actually answer.

Run with:
    python 02_langchain_tool_chain.py

Test it (from the repo root):
    python check.py 02
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# The question we'll ask in BOTH parts. Change the city if you like!
QUESTION = "What notable events or festivals are happening in Delhi this week?"


# -----------------------------------------------------------------------------
# Provided: picks the right LangChain chat model for whichever free provider
# you set up in .env (same idea as project/llm_config.py, LangChain edition).
# -----------------------------------------------------------------------------
def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile").split("/", 1)[-1]
        return ChatGroq(model=model, temperature=0.4)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model = os.getenv(
            "OPENROUTER_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct:free"
        ).split("/", 1)[-1]
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.4,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash").split("/")[-1]
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.4
    )


# -----------------------------------------------------------------------------
# Provided: our "tool" -- a free DuckDuckGo web search, no API key needed.
# A tool is just a plain function that fetches something the LLM can't know.
# -----------------------------------------------------------------------------
def web_search(query: str) -> str:
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No search results found."
    return "\n".join(
        f"{i}. {r.get('title', '')} -- {r.get('body', '')}"
        for i, r in enumerate(results, start=1)
    )


def part_a_plain_chain(llm) -> str:
    """A chain with NO tool -- the model is on its own."""

    # =========================================================================
    # TODO: Student Code Here
    # Build a ChatPromptTemplate that asks the model to answer {question}.
    # Use ChatPromptTemplate.from_template("...your text with {question}...")
    # =========================================================================
    prompt = ChatPromptTemplate.from_template(
    "Answer this question as best you can:\n\n{question}"
    )
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"question": QUESTION})


def part_b_tool_chain(llm) -> str:
    """Same question -- but we search the web FIRST and hand the model the results."""
    search_results = web_search(QUESTION)
    print("\n[web_search returned]\n" + search_results[:500] + "...\n")

    # =========================================================================
    # TODO: Student Code Here
    # Build a ChatPromptTemplate with TWO placeholders: {search_results} and
    # {question}. Tell the model to answer the question USING ONLY the
    # search results (and to cite which result numbers it used).
    # =========================================================================
    prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the search results below. "
    "Cite result numbers.\n\n"
    "SEARCH RESULTS:\n{search_results}\n\nQUESTION: {question}"
)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"search_results": search_results, "question": QUESTION})


def main():
    llm = get_chat_model()

    print("=" * 70)
    print("PART A -- plain chain, no tool (watch it hedge or make things up):")
    print("=" * 70)
    print(part_a_plain_chain(llm).strip())

    print()
    print("=" * 70)
    print("PART B -- same question, but with live web search results:")
    print("=" * 70)
    answer_b = part_b_tool_chain(llm).strip()
    print(answer_b)

    print("\n✅ CHECKPOINT 2 PASSED: your chain answered using live search results.")
    print("   (Compare A and B -- THAT difference is why agents need tools.)")


if __name__ == "__main__":
    main()
