"""
BASICS #4 (Day 2 warm-up): A tiny LangGraph loop (NO LLM, NO API key!)
=======================================================================
Checkpoint: your graph visibly loops at least twice before exiting.

Deliberately NOT the trip planner yet -- this isolates the one idea that
makes LangGraph special: a CONDITIONAL EDGE that can send the flow
BACKWARDS, creating a loop. A chain can't do that. A crew can't do that.

The game: the graph tries to guess a secret number by counting up 1, 2, 3...
  - `make_guess` node  -> guesses the next number
  - `check_guess` node -> compares it to the secret
  - router             -> "again" (loop back) or "done" (exit)   <- YOU write this

This mirrors EXACTLY what you'll do in the trip planner this afternoon,
where the router decides "over budget -> revise" instead of "wrong -> again".

Run with:
    python 04_langgraph_loop.py

Test it (from the repo root):
    python check.py 04
"""

import random
import sys
from typing import TypedDict, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, START, END

MAX_ATTEMPTS = 15  # safety valve -- same idea as MAX_REVISIONS in the trip planner


# --- The shared state: the "clipboard" every node reads and writes -----------
class GameState(TypedDict):
    secret: int      # the number we're trying to guess (4-9)
    guess: int       # the latest guess
    attempts: int    # how many guesses so far


# --- NODE 1: make a guess (counts up: 1, 2, 3, ...) ---------------------------
def make_guess(state: GameState) -> dict:
    new_guess = state["attempts"] + 1
    return {"guess": new_guess, "attempts": state["attempts"] + 1}


# --- NODE 2: check it ----------------------------------------------------------
def check_guess(state: GameState) -> dict:
    verdict = "CORRECT!" if state["guess"] == state["secret"] else "wrong"
    print(f"  Attempt {state['attempts']}: guessed {state['guess']} -> {verdict}")
    return {}  # this node only prints; it changes nothing in the state


# --- THE ROUTER: yours! --------------------------------------------------------
def route_after_check(state: GameState) -> Literal["again", "done"]:
    # =========================================================================
    # TODO: Student Code Here
    # Return "again" if the guess is WRONG and we still have attempts left
    # (attempts < MAX_ATTEMPTS). Otherwise return "done".
    #
    # You have: state["guess"], state["secret"], state["attempts"], MAX_ATTEMPTS
    #
    # (Right now it always returns "done", so the graph never loops --
    #  run it once BEFORE fixing it and watch it give up after 1 guess!)
    # =========================================================================
    return "done"  # TODO: Student Code Here -- replace this line with your condition


def build_game_graph():
    workflow = StateGraph(GameState)

    workflow.add_node("make_guess", make_guess)
    workflow.add_node("check_guess", check_guess)

    workflow.add_edge(START, "make_guess")
    workflow.add_edge("make_guess", "check_guess")

    # THE key line of the whole exercise: a conditional edge that can point
    # BACKWARDS. "again" -> make_guess creates the loop; "done" -> END exits.
    workflow.add_conditional_edges(
        "check_guess",
        route_after_check,
        {"again": "make_guess", "done": END},
    )

    return workflow.compile()


def main():
    secret = random.randint(4, 9)  # always needs >= 4 guesses -> visible loop
    print(f"Secret number is {secret}. Watch the graph loop until it finds it:\n")

    graph = build_game_graph()
    final = graph.invoke(
        {"secret": secret, "guess": 0, "attempts": 0},
        config={"recursion_limit": 100},
    )

    print()
    if final["guess"] == final["secret"] and final["attempts"] >= 3:
        print(f"✅ CHECKPOINT PASSED: your graph looped {final['attempts']} times "
              "before exiting -- that's a real conditional-edge loop.")
    elif final["guess"] != final["secret"]:
        print("✗ NOT THERE YET: the graph stopped before finding the secret.")
        print("  Did you fill in route_after_check()? It still returns 'done' immediately.")
        sys.exit(1)


if __name__ == "__main__":
    main()
