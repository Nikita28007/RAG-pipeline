from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from pro_implementation.security import find_injection_patterns

load_dotenv(override=True)

MODEL = "openai/gpt-5.4-nano"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100
BATCH_SIZE = 2048
PERMANENT_ERRORS = (NotFoundError, AuthenticationError, PermissionDeniedError, BadRequestError)
wait = wait_exponential(multiplier=1, min=10, max=240)

retry_llm = retry(
    wait=wait_exponential(multiplier=1, min=10, max=240),
    stop=stop_after_attempt(5),
    retry=retry_if_not_exception_type(PERMANENT_ERRORS),
    reraise=True,
)

WORKERS = 3

openai = OpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry_llm
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

class IngestError(RuntimeError):
    """Plain, picklable error that carries the original failure across processes."""

def process_document_safe(document):
    """Worker entry point.

    litellm exceptions cannot be unpickled in the parent process (their __init__ needs
    extra arguments), which hides the real error behind a TypeError. Converting the
    failure to a plain error keeps the original message visible.
    """
    try:
        return process_document(document)
    except Exception as e:
        raise IngestError(f"{document['source']}: {type(e).__name__}: {e}") from None


def preflight():
    """Fail fast with the real error before paying for every document."""
    try:
        completion(model=MODEL, messages=[{"role": "user", "content": "ping"}])
    except Exception as e:
        raise SystemExit(
            f"Chunking model '{MODEL}' is not usable: {type(e).__name__}: {e}\n"
            "Set INGEST_MODEL to a model your API key can access."
        ) from None
    try:
        openai.embeddings.create(model=embedding_model, input=["ping"])
    except Exception as e:
        raise SystemExit(
            f"Embedding model '{embedding_model}' is not usable: {type(e).__name__}: {e}"
        ) from None

def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document_safe, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def report_suspicious_chunks(chunks):
    """Print chunks that match prompt-injection heuristics. Heuristics can be bypassed."""
    suspicious = []
    for chunk in chunks:
        patterns = find_injection_patterns(chunk.page_content)
        if patterns:
            suspicious.append((chunk.metadata["source"], patterns))
            print(f"WARNING possible prompt injection in {chunk.metadata['source']}: {patterns}")
    return suspicious

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]

    
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_emb = openai.embeddings.create(model=embedding_model, input=batch).data
        all_embeddings.extend(batch_emb)
    emb = all_embeddings

    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    preflight()
    documents = fetch_documents()
    chunks = create_chunks(documents)
    report_suspicious_chunks(chunks)
    create_embeddings(chunks)
    print("Ingestion complete")
