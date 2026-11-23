"""LangChain tools for the agents. All file access goes through the sandboxed Workspace.

There is deliberately no tool for shell commands, environment variables or network access.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from pocketdev_agent.sandbox import Workspace

READ_ONLY_TOOLS = {"read_file", "list_dir", "search_text"}


def make_tools(ws: Workspace, *, read_only: bool = False) -> list[BaseTool]:
    def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
        """Read a text file from the project. `path` is relative to the project root.
        Returns numbered lines (the numbers are not part of the file). Use start_line to page through long files."""
        return ws.read_file(path, start_line, max_lines)

    def list_dir(path: str = "") -> str:
        """List files and folders in a project folder (relative path, "" for the project root). Folders end with '/'."""
        return ws.list_dir(path)

    def search_text(query: str, path: str = "") -> str:
        """Case-insensitive plain-text search across project files (optionally only under folder `path`).
        Returns 'file:line: text' matches."""
        return ws.search_text(query, path)

    def write_file(path: str, content: str) -> str:
        """Create or completely overwrite a text file in the project with `content` (the full new file)."""
        return ws.write_file(path, content)

    def replace_in_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact, unique occurrence of `old_text` with `new_text` in an existing file.
        Copy old_text exactly from read_file output (without the line-number prefix)."""
        return ws.replace_in_file(path, old_text, new_text)

    def delete_file(path: str) -> str:
        """Delete a file from the project. Only do this when the plan says so."""
        return ws.delete_file(path)

    fns = [read_file, list_dir, search_text]
    if not read_only:
        fns += [write_file, replace_in_file, delete_file]
    return [StructuredTool.from_function(fn) for fn in fns]


def describe_call(name: str, args: dict) -> str:
    """Plain-language progress line for a tool call (shown to the user)."""
    path = str(args.get("path") or "").strip() or "the project"
    return {
        "read_file": f"Reading {path}",
        "list_dir": f"Looking around {path}",
        "search_text": f"Searching for \"{str(args.get('query', ''))[:60]}\"",
        "write_file": f"Writing {path}",
        "replace_in_file": f"Editing {path}",
        "delete_file": f"Deleting {path}",
    }.get(name, f"Using {name}")
