# Basics — Hands-on Templates (Day 1 + Day 2 warm-up)

Work through these in order. Each file tells you exactly what to fill in
(search for `TODO: Student Code Here`) and prints a `✅ CHECKPOINT PASSED`
line when it works.

| # | File | You build | Needs API key? |
|---|---|---|---|
| 1 | `01_first_api_call.py` | A raw HTTPS call to an LLM — no framework | Yes |
| 2 | `02_langchain_tool_chain.py` | A LangChain chain, then a search-tool-augmented one | Yes |
| 3 | `03_first_crew.py` | A 2-agent CrewAI crew (Researcher → Writer) | Yes |
| 4 | `04_langgraph_loop.py` | A LangGraph loop with a conditional edge | **No** — runs free & instantly |

## Setup (once)

```bash
# from the repo root:
pip install -r project/requirements.txt
cp basics/.env.example basics/.env    # then paste your free API key into it
```

## Run each file — just the filename

```bash
cd basics
python 01_first_api_call.py
python 02_langchain_tool_chain.py
python 03_first_crew.py
python 04_langgraph_loop.py
```

## Test any file by its name

From the repo root — this is how you prove a checkpoint is done:

```bash
python check.py 01          # or: python check.py 01_first_api_call.py
python check.py 02
python check.py 03
python check.py 04          # runs for real — it has no LLM inside
python check.py basics      # static check of all four at once
```

`check.py <name>` first verifies you filled every TODO, then actually runs
the file and looks for its `✅ CHECKPOINT PASSED` line.
