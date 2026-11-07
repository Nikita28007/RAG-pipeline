from pathlib import Path

from evaluation.test import load_tests

KB = Path(__file__).resolve().parent.parent / "knowledge-base"


def test_eval_set_is_well_formed():
    tests = load_tests()
    assert tests
    assert all(t.keywords and t.reference_answer.strip() and t.category for t in tests)


def test_eval_questions_are_unique():
    questions = [t.question for t in load_tests()]
    assert len(questions) == len(set(questions))


def test_knowledge_base_has_documents_in_every_type_folder():
    folders = [f for f in KB.iterdir() if f.is_dir()]
    assert folders
    for folder in folders:
        assert list(folder.rglob("*.md")), f"{folder.name} has no documents"