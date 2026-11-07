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


GRADIO6_HISTORY = [
    {"role": "user", "metadata": None, "options": None, "content": [{"type": "text", "text": "hello"}]},
    {"role": "assistant", "metadata": None, "options": None, "content": [{"type": "text", "text": "hi"}]},
    {"role": "user", "metadata": None, "options": None, "content": [{"type": "text", "text": "who founded it?"}]},
]


def test_chat_appends_assistant_message_and_context(monkeypatch):
    import app

    monkeypatch.setattr(app, "answer_question", lambda question, history: ("the answer", []))
    history, context = app.chat([{"role": "user", "content": "hi"}])
    assert history[-1] == {"role": "assistant", "content": "the answer"}
    assert "Relevant Context" in context


def test_chat_passes_plain_text_to_the_rag_pipeline(monkeypatch):
    import app

    seen = {}

    def fake_answer(question, history):
        seen["question"], seen["history"] = question, history
        return "the answer", []

    monkeypatch.setattr(app, "answer_question", fake_answer)
    history, _ = app.chat(GRADIO6_HISTORY)
    assert seen["question"] == "who founded it?"
    assert seen["history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert history[-1] == {"role": "assistant", "content": "the answer"}


def test_message_text_handles_all_content_shapes():
    import app

    assert app.message_text("plain") == "plain"
    assert app.message_text({"type": "text", "text": "block"}) == "block"
    assert app.message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_gradio_chatbot_output_works_with_chat(monkeypatch):
    """Feed chat() exactly what the Gradio 6 Chatbot component hands to handlers."""
    import app
    from gradio.components import chatbot as gr_chatbot

    payload = gr_chatbot.ChatbotDataMessages(
        root=[gr_chatbot.Message(role="user", content=[{"type": "text", "text": "who founded it?"}])]
    )
    handler_input = gr.Chatbot().preprocess(payload)
    monkeypatch.setattr(app, "answer_question", lambda question, history: (question.upper(), []))
    history, _ = app.chat(handler_input)
    assert history[-1]["content"] == "WHO FOUNDED IT?"