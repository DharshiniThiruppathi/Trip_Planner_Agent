"""
BASICS #3 (Day 1): Your first CrewAI crew (Researcher + Writer)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

TOPIC = "the latest developments in electric vehicles in India"


def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        return LLM(
            model=os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.4,
        )

    if provider == "openrouter":
        return LLM(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            ),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.4,
        )

    return LLM(
        model=os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )


@tool("Web Search")
def web_search(query: str) -> str:
    """
    Searches the web and returns the top 5 results.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No results found."

    return "\n".join(
        f"{i}. {r.get('title','')} -- {r.get('body','')}"
        for i, r in enumerate(results, start=1)
    )


def main():
    llm = get_llm()

    # -----------------------------
    # Research Agent
    # -----------------------------
    researcher = Agent(
        role="Internet Researcher",
        goal=f"Find 3-5 current, concrete facts about {TOPIC}, with sources.",
        backstory=(
            "You are a meticulous researcher who verifies every fact "
            "using live web search. You always include names, numbers, "
            "dates, and mention the source."
        ),
        tools=[web_search],
        llm=llm,
        verbose=True,
    )

    # -----------------------------
    # Writer Agent
    # -----------------------------
    writer = Agent(
        role="Tech Writer",
        goal="Turn the Researcher's findings into a crisp 3-sentence summary that anyone can understand.",
        backstory=(
            "You are a professional technology journalist. "
            "You never invent facts. "
            "You only rewrite the Researcher's findings into clear, "
            "engaging language while preserving important numbers."
        ),
        llm=llm,
        verbose=True,
    )

    # -----------------------------
    # Research Task
    # -----------------------------
    research_task = Task(
        description=(
            f"""
Research the latest developments in {TOPIC}.

Use the Web Search tool at least once.

Collect 3-5 current facts.

Each fact should include:
- numbers if available
- names
- dates
- source
"""
        ),
        expected_output="A bullet list of 3-5 researched facts with sources.",
        agent=researcher,
    )

    # -----------------------------
    # Writer Task
    # -----------------------------
    write_task = Task(
        description=(
            """
Take the Researcher's findings and produce EXACTLY THREE sentences.

Requirements:
- Plain English
- Keep important numbers
- Do not invent facts
- Do not add new information
"""
        ),
        expected_output="Exactly three sentences.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    main()