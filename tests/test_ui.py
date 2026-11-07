import gradio as gr
import pytest


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    launched = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: launched.update(kwargs))
    return launched


@pytest.mark.parametrize("module_name", ["app", "evaluator"])
def test_ui_builds_without_launching(module_name, no_launch):
    module = __import__(module_name)
    module.main()
    assert "theme" in no_launch


def test_chat_appends_assistant_message_and_context(monkeypatch):
    import app

    monkeypatch.setattr(app, "answer_question", lambda question, history: ("the answer", []))
    history, context = app.chat([{"role": "user", "content": "hi"}])
    assert history[-1] == {"role": "assistant", "content": "the answer"}
    assert "Relevant Context" in context