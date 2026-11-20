import ast
from pathlib import Path


def test_scripts_parse():
    for f in (Path(__file__).resolve().parents[2] / "scripts").glob("*.py"):
        ast.parse(f.read_text())
