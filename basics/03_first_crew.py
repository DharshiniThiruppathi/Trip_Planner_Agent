"""
BASICS #3 (Day 1): Your first CrewAI crew (Researcher + Writer)
================================================================
Checkpoint: the crew runs end-to-end and prints a coherent 3-sentence
summary that clearly used the Researcher's live findings.

The film crew analogy: instead of one giant prompt doing everything badly,
you hire specialists. A RESEARCHER digs up facts with a search tool; a
WRITER turns those facts into clean prose. Each gets a role, a goal, and a
backstory -- specificity is what makes their output focused instead of
generic.

The Researcher below is COMPLETE -- it's your worked example.
The Writer is YOURS to fill in.

Run with:
    python 03_first_crew.py

Test it (from the repo root):
    python check.py 03
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# The topic our crew will research + summarize. Change it!
TOPIC = "the latest developments in electric vehicles in India"


# -----------------------------------------------------------------------------
# Provided: LLM setup (same pattern as project/llm_config.py).
# -----------------------------------------------------------------------------
def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        return LLM(model=os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
                   api_key=os.getenv("GROQ_API_KEY"), temperature=0.4)
    if provider == "openrouter":
        return LLM(model=os.getenv("OPENROUTER_MODEL",
                                   "openrouter/meta-llama/llama-3.3-70b-instruct:free"),
                   api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.4)
    return LLM(model=os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash"),
               api_key=os.getenv("GEMINI_API_KEY"), temperature=0.4)


# -----------------------------------------------------------------------------
# Provided: a web search tool. In CrewAI, the @tool decorator turns any
# plain function into something an agent can decide to call. The docstring
# is NOT decoration -- it's how the agent knows when to use the tool!
# -----------------------------------------------------------------------------
@tool("Web Search")
def web_search(query: str) -> str:
    """Searches the live web and returns the top 5 results with titles and
    snippets. Use this to find current facts you don't already know."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    return "\n".join(
        f"{i}. {r.get('title', '')} -- {r.get('body', '')}"
        for i, r in enumerate(results, start=1)
    )


def main():
    llm = get_llm()

    # -- AGENT 1: the Researcher (COMPLETE -- study this shape) ---------------
    researcher = Agent(
        role="Internet Researcher",
        goal=f"Find 3-5 current, concrete facts about {TOPIC}, with sources.",
        backstory=(
            "You are a meticulous researcher who never states a fact without "
            "having seen it in a search result first. You love recent, "
            "specific details -- numbers, names, dates -- and you always note "
            "which search result each fact came from."
        ),
        tools=[web_search],
        llm=llm,
        verbose=True,
    )

    # -- AGENT 2: the Writer (YOURS) ------------------------------------------
    writer = Agent(
        role="Tech Writer",
        # =====================================================================
        # TODO: Student Code Here
        # ONE sentence: what does the Writer produce? (Hint: a crisp,
        # 3-sentence summary a busy person could read in 15 seconds.)
        # =====================================================================
        goal="TODO: Student Code Here",
        # =====================================================================
        # TODO: Student Code Here
        # 2-3 sentences of personality. Who are they? What do they refuse to
        # do (jargon? fluff? unverified claims?)
        # =====================================================================
        backstory="TODO: Student Code Here",
        llm=llm,  # note: the Writer gets NO tools -- writing needs no search
        verbose=True,
    )

    # -- TASK 1: research (COMPLETE) -------------------------------------------
    research_task = Task(
        description=(
            f"Research: {TOPIC}. Use the Web Search tool at least once. "
            "Collect 3-5 current, concrete facts (numbers, names, dates)."
        ),
        expected_output="A bullet list of 3-5 facts, each with its source noted.",
        agent=researcher,
    )

    # -- TASK 2: write (YOURS) --------------------------------------------------
    write_task = Task(
        # =========================================================================
        # TODO: Student Code Here
        # Tell the Writer what to do with the Researcher's findings:
        # turn them into a summary of EXACTLY 3 sentences, plain language,
        # keeping the most interesting numbers.
        # =========================================================================
        description="TODO: Student Code Here",
        expected_output="Exactly 3 sentences of plain, engaging prose.",
        agent=writer,
        context=[research_task],  # <-- this hands the Researcher's output to the Writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,  # researcher first, then writer
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY FROM YOUR CREW:")
    print("=" * 70)
    print(str(result).strip())
    print("\n✅ CHECKPOINT 3 PASSED: your 2-agent crew ran end-to-end.")
    print("   (Scroll up: can you see the Researcher's findings feeding the Writer?)")


if __name__ == "__main__":
    main()
