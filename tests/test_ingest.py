from pro_implementation import ingest


def test_fetch_documents_reads_every_markdown_file():
    documents = ingest.fetch_documents()
    assert documents
    assert all({"type", "source", "text"} <= d.keys() and d["text"] for d in documents)


def test_make_prompt_contains_document_metadata():
    document = {"type": "company", "source": "kb/company/about.md", "text": "x" * 450}
    prompt = ingest.make_prompt(document)
    assert "company" in prompt and "kb/company/about.md" in prompt
    assert "at least 5 chunks" in prompt  # 450 // 100 + 1


def test_chunk_as_result_embeds_headline_summary_and_text():
    chunk = ingest.Chunk(headline="Head", summary="Sum", original_text="Text")
    result = chunk.as_result({"type": "company", "source": "kb/a.md"})
    assert result.page_content == "Head\n\nSum\n\nText"
    assert result.metadata == {"source": "kb/a.md", "type": "company"}