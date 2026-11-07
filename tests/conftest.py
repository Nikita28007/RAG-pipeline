import os
import sys
import types

# The pro_implementation modules create an OpenAI client at import time.
# A dummy key is enough: no network calls are made by these tests.
os.environ.setdefault("OPENAI_API_KEY", "ci-dummy-key")

# implementation.answer loads a HuggingFace embedding model at import, which
# needs torch and a model download. Tests only need the metric functions in
# evaluation.eval, so the retrieval module is replaced by a stub.
_stub = types.ModuleType("implementation.answer")
_stub.answer_question = lambda question, history=None: ("", [])
_stub.fetch_context = lambda question: []
sys.modules.setdefault("implementation.answer", _stub)