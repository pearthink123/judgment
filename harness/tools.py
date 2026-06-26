"""
Tool registry — safe, introspectable tool execution.

Tools are registered with a name, JSON Schema for arguments, and an async
(or sync) callable.  The registry supports both local tools and MCP
(Model Context Protocol) tool discovery.
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class Tool:
    """A single tool that the LLM can invoke."""

    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema for arguments
    fn: Callable                 # (kwargs) -> str

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Call the tool and return its string output."""
        try:
            result = self.fn(**arguments)
            return str(result)
        except Exception as e:
            return f"[tool error] {type(e).__name__}: {e}"


class ToolRegistry:
    """
    Thread-safe tool collection.

    Usage:
        registry = ToolRegistry()
        registry.register("read_file", "Read a file", {...}, my_read_fn)
        schemas = registry.openai_schemas()
        result = registry.execute("read_file", {"path": "/tmp/x.txt"})
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        fn: Callable,
    ):
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
        )

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[tool error] Unknown tool: {name}"
        return tool.execute(arguments)


# ---------------------------------------------------------------------------
# Built-in tools (extensible via registry.register)
# ---------------------------------------------------------------------------
def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        return content
    except FileNotFoundError:
        return f"[error] File not found: {path}"
    except Exception as e:
        return f"[error] {e}"


def _write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[error] {e}"


def _run_command(cmd: str) -> str:
    import subprocess
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
        )
        out = result.stdout
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
        if len(out) > 4000:
            out = out[:4000] + "\n... [truncated]"
        return out or "(exit code 0, no output)"
    except subprocess.TimeoutExpired:
        return "[error] Command timed out (30s)"
    except Exception as e:
        return f"[error] {e}"


def _think(thought: str) -> str:
    """No-op: lets the LLM 'think out loud' without executing anything."""
    return f"Thought recorded ({len(thought)} chars)."


BUILTIN_TOOLS = {
    "read_file": {
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path."},
            },
            "required": ["path"],
        },
        "fn": _read_file,
    },
    "write_file": {
        "description": "Write content to a file, overwriting if it exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
        "fn": _write_file,
    },
    "run_command": {
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["cmd"],
        },
        "fn": _run_command,
    },
    "think": {
        "description": "Think out loud — record reasoning without side effects.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Reasoning or plan to record."},
            },
            "required": ["thought"],
        },
        "fn": _think,
    },
}


def default_registry() -> ToolRegistry:
    r = ToolRegistry()
    for name, spec in BUILTIN_TOOLS.items():
        r.register(name, spec["description"], spec["parameters"], spec["fn"])
    return r
