from __future__ import annotations

import ast
from typing import Any

from smolagents import LocalPythonExecutor, PythonExecutor


class _RenameSendMessage(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "send_message":
            node.func.id = "final_answer"
        return node


def _rewrite_terminal_call(code: str) -> str:
    """Rewrite only ``send_message(...)`` to smolagents' native terminal primitive."""
    tree = ast.parse(code)
    tree = _RenameSendMessage().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class TerminalExecutor(PythonExecutor):
    """Expose ``send_message`` while reusing smolagents' final-answer machinery."""

    def __init__(self, executor: PythonExecutor):
        self.executor = executor

    def send_variables(self, variables: dict[str, Any]) -> None:
        self.executor.send_variables(variables)

    def send_tools(self, tools: dict[str, Any]) -> None:
        mapped = dict(tools)
        terminal = mapped.pop("send_message", None)
        if terminal is None:
            raise ValueError("send_message tool is required")
        mapped.pop("final_answer", None)
        # LocalPythonExecutor treats a tool named final_answer as the terminal primitive.
        mapped["final_answer"] = terminal
        self.executor.send_tools(mapped)

    def __call__(self, code_action: str):
        return self.executor(_rewrite_terminal_call(code_action))

    @property
    def state(self) -> dict[str, Any]:
        return getattr(self.executor, "state", {})

    def cleanup(self) -> None:
        cleanup = getattr(self.executor, "cleanup", None)
        if cleanup is not None:
            cleanup()


def local_executor(*, timeout_seconds: int | None = 30) -> TerminalExecutor:
    return TerminalExecutor(
        LocalPythonExecutor(
            additional_authorized_imports=[],
            timeout_seconds=timeout_seconds,
        )
    )
