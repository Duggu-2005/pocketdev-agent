"""PocketDev AI agent runner.

Runs inside a GitHub Actions workflow in the user's own repository: fetches the task from the
PocketDev API, runs the LangGraph agents (planner, coder, tester, reviewer, summarizer), pushes a
feature branch and reports back. See README.md.
"""

__version__ = "0.1.0"
