#!/usr/bin/env python3
"""Prove a comment/docstring cleanup changed NO code: every touched file's AST
must be identical to its base version. Docstrings are AST nodes, so they are
normalized out explicitly -- that is the only intended difference."""
import ast, subprocess, sys


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def main(base: str) -> int:
    files = subprocess.run(["git", "diff", "--name-only", base],
                           capture_output=True, text=True).stdout.split()
    bad = []
    for f in files:
        if not f.endswith(".py"):
            continue
        before = subprocess.run(["git", "show", f"{base}:{f}"],
                                capture_output=True, text=True).stdout
        after = open(f).read()
        a = ast.dump(strip_docstrings(ast.parse(before)))
        b = ast.dump(strip_docstrings(ast.parse(after)))
        status = "OK  " if a == b else "CODE CHANGED"
        if a != b:
            bad.append(f)
        print(f"  {status} {f}")
    print(f"\n{len(files)} file(s); {len(bad)} with code changes")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "ghhttps/master"))
