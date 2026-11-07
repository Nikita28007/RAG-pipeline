from pathlib import Path

KB = Path(__file__).resolve().parent.parent / "knowledge-base"


def knowledge_base_texts():
    for path in KB.rglob("*.md"):
        yield path.as_posix(), path.read_text(encoding="utf-8")