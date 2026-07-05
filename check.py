"""
check.py -- Workshop checkpoint checker.
=========================================
Lets students (and the instructor) verify every checkpoint WITHOUT burning
LLM calls: everything here is offline/static unless you pass --live or
test a basics file by name (which runs it for real).

Usage (from the repo root):
    python check.py                      # everything, offline
    python check.py setup                # environment + packages + API key present
    python check.py basics               # all basics templates: TODOs left? syntax ok?
    python check.py 01                   # ONE basics file by name: TODO check + REAL run
    python check.py 03_first_crew.py     # (same thing, full filename works too)
    python check.py capstone             # the trip planner (YAML, router, tools, graph)
    python check.py result               # same capstone checks against project_result/
    python check.py all --live           # the works, running every basics file live

What "offline" can still genuinely test in the capstone:
  - the router functions (pure Python -- fed fake states, no LLM involved)
  - the expense calculator's arithmetic
  - the currency converter's fallback math
  - that YAML TODOs are filled and required {placeholders} survived
  - that the LangGraph graph actually compiles
The only thing it can't judge offline is prose quality -- for that, run
`python project/main.py` once at the end.
"""

import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
BASICS = os.path.join(ROOT, "basics")
PROJECT = os.path.join(ROOT, "project")
PROJECT_RESULT = os.path.join(ROOT, "project_result")

TODO_MARKER = "TODO: Student Code Here"
PLACEHOLDER_KEY_HINTS = ("your-gemini-api-key-here", "your-groq-api-key-here",
                         "your-openrouter-api-key-here", "")

BASICS_FILES = [
    "01_first_api_call.py",
    "02_langchain_tool_chain.py",
    "03_first_crew.py",
    "04_langgraph_loop.py",
]

RESULTS = []  # (status, section, message)


def record(ok, section, message, warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((status, section, message))
    print(f"  [{status}] {message}")


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def todo_lines(path):
    """Returns [(line_number, line)] for every student-TODO still in the file.

    Pure comment lines are ignored: the instruction blocks above each blank
    stay in the file forever -- only markers on actual CODE lines (like
    `PROMPT = "TODO: ..."` or `prompt = None  # TODO ...`) mean unfinished work.
    """
    hits = []
    for i, line in enumerate(read_text(path).splitlines(), start=1):
        if TODO_MARKER in line and not line.strip().startswith("#"):
            hits.append((i, line.strip()))
    return hits


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
def check_setup(live=False):
    section("SETUP -- environment & keys")

    ok = sys.version_info >= (3, 10)
    record(ok, "setup", f"Python {sys.version.split()[0]} (need 3.10+)")

    for pkg, why in [
        ("dotenv", "python-dotenv (reads .env)"),
        ("requests", "requests (raw API calls, weather & currency tools)"),
        ("yaml", "pyyaml (agents.yaml / tasks.yaml)"),
        ("ddgs", "ddgs (free DuckDuckGo search)"),
        ("langchain_core", "langchain-core (basics chains)"),
        ("crewai", "crewai (agents)"),
        ("langgraph", "langgraph (the state machine)"),
        ("streamlit", "streamlit (the UI)"),
    ]:
        try:
            __import__(pkg)
            record(True, "setup", f"import {pkg} -- {why}")
        except ImportError as exc:
            record(False, "setup", f"import {pkg} FAILED ({exc}) -- pip install -r project/requirements.txt")

    # .env can live in repo root, basics/, or project/
    env_paths = [os.path.join(d, ".env") for d in (ROOT, BASICS, PROJECT)]
    found = [p for p in env_paths if os.path.exists(p)]
    record(bool(found), "setup",
           f".env file found at: {found[0]}" if found
           else ".env not found (copy basics/.env.example or project/.env.example to .env)")

    if found:
        from dotenv import load_dotenv
        load_dotenv(found[0])
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        key_name = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY"}.get(provider)
        if key_name is None:
            record(False, "setup", f"LLM_PROVIDER='{provider}' is not one of gemini/groq/openrouter")
        else:
            key = (os.getenv(key_name) or "").strip()
            ok = key not in PLACEHOLDER_KEY_HINTS
            record(ok, "setup",
                   f"{key_name} is set for provider '{provider}'" if ok
                   else f"{key_name} is missing or still the placeholder -- paste your real key into .env")

            if live and ok:
                _live_ping(provider, key)


def _live_ping(provider, key):
    """One tiny real API call to prove the key actually works."""
    import requests
    try:
        if provider == "gemini":
            model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash").split("/")[-1]
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": key},
                json={"contents": [{"parts": [{"text": "Reply with the single word: pong"}]}]},
                timeout=30)
        else:
            url = ("https://api.groq.com/openai/v1/chat/completions" if provider == "groq"
                   else "https://openrouter.ai/api/v1/chat/completions")
            model_env = "GROQ_MODEL" if provider == "groq" else "OPENROUTER_MODEL"
            default = ("groq/llama-3.3-70b-versatile" if provider == "groq"
                       else "openrouter/meta-llama/llama-3.3-70b-instruct:free")
            model = os.getenv(model_env, default).split("/", 1)[-1]
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"},
                              json={"model": model,
                                    "messages": [{"role": "user",
                                                  "content": "Reply with the single word: pong"}]},
                              timeout=30)
        record(r.status_code == 200, "setup",
               f"LIVE ping to {provider}: HTTP {r.status_code}"
               + ("" if r.status_code == 200 else f" -- {r.text[:150]}"))
    except Exception as exc:  # noqa: BLE001
        record(False, "setup", f"LIVE ping to {provider} failed: {exc}")


# ---------------------------------------------------------------------------
# BASICS
# ---------------------------------------------------------------------------
def check_basics_file(fname, run_it):
    """Static check of one basics file; optionally runs it for real."""
    path = os.path.join(BASICS, fname)
    if not os.path.exists(path):
        record(False, "basics", f"{fname} is missing")
        return

    # 1. Does it still parse as valid Python?
    try:
        compile(read_text(path), fname, "exec")
        record(True, "basics", f"{fname}: valid Python syntax")
    except SyntaxError as exc:
        record(False, "basics", f"{fname}: SYNTAX ERROR at line {exc.lineno}: {exc.msg}")
        return

    # 2. Any TODOs left on code lines?
    hits = todo_lines(path)
    if hits:
        lines = ", ".join(str(n) for n, _ in hits)
        record(False, "basics", f"{fname}: {len(hits)} TODO(s) left to fill (line {lines})")
    else:
        record(True, "basics", f"{fname}: all TODOs filled in")

    # 3. Run it? (04 has no LLM inside, so running it is always free.)
    if run_it and not hits:
        _run_script(path, cwd=BASICS, section_name="basics", timeout=240)
    elif run_it and hits:
        record(False, "basics", f"{fname}: skipping the live run until the TODOs are filled")


def check_basics(live=False):
    section("BASICS -- hands-on templates")
    for fname in BASICS_FILES:
        # 04 is free to run (no LLM), so always run it; others only with --live.
        run_it = live or fname == "04_langgraph_loop.py"
        check_basics_file(fname, run_it=run_it)


def _run_script(path, cwd, section_name, timeout):
    fname = os.path.basename(path)
    print(f"  ... running {fname} for real (this may take a minute)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, fname], cwd=cwd, env=env,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        passed = "CHECKPOINT" in proc.stdout and "PASSED" in proc.stdout
        if passed:
            record(True, section_name, f"{fname}: LIVE RUN passed its checkpoint")
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
            record(False, section_name,
                   f"{fname}: live run did NOT print its checkpoint line. Last output: "
                   + " | ".join(tail))
    except subprocess.TimeoutExpired:
        record(False, section_name, f"{fname}: timed out after {timeout}s")


# ---------------------------------------------------------------------------
# CAPSTONE (the trip planner) -- entirely offline
# ---------------------------------------------------------------------------
def check_capstone(project_dir=PROJECT, label="CAPSTONE"):
    section(f"{label} -- config/agents.yaml & config/tasks.yaml")
    import yaml

    if not os.path.isdir(project_dir):
        record(False, "capstone", f"folder not found: {project_dir}")
        return

    agents = yaml.safe_load(read_text(os.path.join(project_dir, "config", "agents.yaml")))
    tasks = yaml.safe_load(read_text(os.path.join(project_dir, "config", "tasks.yaml")))

    for agent_id in ("local_guide", "concierge"):
        for field in ("goal", "backstory"):
            value = str(agents.get(agent_id, {}).get(field, ""))
            ok = TODO_MARKER not in value and len(value.strip()) > 20
            record(ok, "capstone",
                   f"agents.yaml -> {agent_id}.{field}: "
                   + ("filled in" if ok else "still a TODO (or too short)"))

    required_placeholders = {
        "itinerary_task": {
            "description": ["{selected_bases}", "{trip_length_days}", "{expectations}"],
            "expected_output": [],
        },
        "budget_task": {
            "description": ["{itinerary}", "{budget_inr}", "{selected_bases}"],
            "expected_output": ["TOTAL_COST_INR"],
        },
    }
    for task_id, fields in required_placeholders.items():
        for field, needles in fields.items():
            value = str(tasks.get(task_id, {}).get(field, ""))
            if TODO_MARKER in value or len(value.strip()) <= 20:
                record(False, "capstone", f"tasks.yaml -> {task_id}.{field}: still a TODO")
                continue
            missing = [n for n in needles if n not in value]
            record(not missing, "capstone",
                   f"tasks.yaml -> {task_id}.{field}: "
                   + ("filled in, all required pieces present" if not missing
                      else f"missing required {missing} -- graph_workflow.py .format() "
                           "fills these at runtime"))

    section(f"{label} -- tools & routers (behavior tests, no LLM)")

    os.environ["CURRENCY_TOOL_OFFLINE"] = "1"  # never hit the network from a checker

    # Import the target folder's graph_workflow fresh (drop any cached copy
    # from a previous check_capstone call against a different folder).
    for mod in list(sys.modules):
        if mod in ("graph_workflow", "llm_config", "tools") or mod.startswith("tools."):
            del sys.modules[mod]
    while project_dir in sys.path:
        sys.path.remove(project_dir)
    sys.path.insert(0, project_dir)
    try:
        import graph_workflow as gw
    except Exception as exc:  # noqa: BLE001
        record(False, "capstone", f"could not import {project_dir}/graph_workflow.py: {exc}")
        return

    # --- Expense calculator: exact arithmetic (multi-city route sample) ---
    out = gw.expense_tool._run(
        num_days=5, num_travelers=1, travel_fare_per_person=3000,
        intercity_transport_total=1000, stay_cost_per_night=1500,
        food_cost_per_day_per_person=600, activities_total=2000,
        local_transport_per_day=500, contingency_percent=10,
    )
    # travel 3000 + intercity 1000 + stay 7500 + food 3000 + local 2500
    # + activities 2000 = 19000 subtotal; +10% -> 20900
    record("20,900" in out, "capstone",
           "ExpenseCalculatorTool arithmetic: 5-day 2-base sample totals INR 20,900"
           + ("" if "20,900" in out else f" -- got:\n{out}"))

    # --- Currency converter: offline fallback math ---
    out = gw.currency_tool._run(amount=100, from_currency="USD", to_currency="INR")
    record("8,800.00" in out, "capstone",
           "CurrencyConverterTool offline fallback: 100 USD -> 8,800.00 INR (approx table)"
           + ("" if "8,800.00" in out else f" -- got: {out}"))

    # --- Worked-example router (provided code -- should always pass) ---
    ok1 = gw.route_after_reality_check({"budget_verdict": "UNREALISTIC"}) == "advise"
    ok2 = gw.route_after_reality_check({"budget_verdict": "REALISTIC"}) == "plan"
    record(ok1 and ok2, "capstone", "route_after_reality_check (worked example) behaves correctly")

    # --- THE student router: 3 scenarios ---
    over_budget = {"estimated_total_cost_inr": 60000.0, "budget_inr": 50000.0,
                   "revision_count": 0}
    under_budget = {"estimated_total_cost_inr": 40000.0, "budget_inr": 50000.0,
                    "revision_count": 0}
    out_of_retries = {"estimated_total_cost_inr": 60000.0, "budget_inr": 50000.0,
                      "revision_count": gw.MAX_REVISIONS}

    got = gw.route_after_budget_check(over_budget)
    record(got == "revise", "capstone",
           "router: OVER budget with retries left -> 'revise'"
           + ("" if got == "revise" else
              f" (got '{got}'{' -- the TODO is not filled in yet' if got == 'end' else ''})"))

    got = gw.route_after_budget_check(under_budget)
    record(got == "end", "capstone",
           f"router: UNDER budget -> 'end'" + ("" if got == "end" else f" (got '{got}')"))

    got = gw.route_after_budget_check(out_of_retries)
    record(got == "end", "capstone",
           "router: over budget but MAX_REVISIONS reached -> 'end' (loop must terminate!)"
           + ("" if got == "end" else f" (got '{got}')"))

    # --- Does the whole graph compile? ---
    try:
        gw.build_graph()
        record(True, "capstone", "build_graph(): the LangGraph state machine compiles")
    except Exception as exc:  # noqa: BLE001
        record(False, "capstone", f"build_graph() failed: {exc}")

    print("\n  Tip: for the full end-to-end test (real LLM calls), run:")
    print(f"       cd {os.path.basename(project_dir)} && python main.py")


# ---------------------------------------------------------------------------
def resolve_basics_target(target):
    """'01' or '03_first_crew.py' -> the matching basics filename, or None."""
    name = target if target.endswith(".py") else target
    for fname in BASICS_FILES:
        if fname == name or fname.startswith(name):
            return fname
    return None


def main():
    args = [a for a in sys.argv[1:]]
    live = "--live" in args
    targets = [a for a in args if not a.startswith("--")] or ["all"]
    target = targets[0].lower()

    basics_file = resolve_basics_target(target)

    if basics_file:
        # Test ONE basics file by name: static check + real run.
        section(f"BASICS -- {basics_file}")
        check_basics_file(basics_file, run_it=True)
    elif target in ("setup", "basics", "capstone", "result", "all"):
        if target in ("setup", "all"):
            check_setup(live=live)
        if target in ("basics", "all"):
            check_basics(live=live)
        if target in ("capstone", "all"):
            check_capstone(PROJECT, label="CAPSTONE (project/)")
        if target == "result":
            check_capstone(PROJECT_RESULT, label="RESULT (project_result/)")
    else:
        print(__doc__)
        sys.exit(2)

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    passes = [r for r in RESULTS if r[0] == "PASS"]
    print("\n" + "=" * 66)
    print(f"SUMMARY: {len(passes)} passed, {len(fails)} failed")
    if fails:
        print("Still to fix:")
        for _, sec, msg in fails:
            print(f"  - [{sec}] {msg}")
    else:
        print("Everything checked out. Go run the real thing!")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
