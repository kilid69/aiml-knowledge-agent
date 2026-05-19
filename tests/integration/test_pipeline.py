"""Integration test for IngestionPipeline — real LM Studio + real Qdrant.

The unit test (`tests/unit/test_pipeline.py`) covers orchestration logic
with fakes. This test verifies the full stack actually works: a real
chunker chunks a real string, real embeddings come out of LM Studio, and
the points land in Qdrant with the right shape.

Uses the `isolated_collection` fixture so the production collection is
never touched.
"""

from aiml_knowledge_agent.ingestion.chunkers.simple import SimpleChunker
from aiml_knowledge_agent.ingestion.embedder import Embedder
from aiml_knowledge_agent.ingestion.pipeline import IngestionPipeline
from aiml_knowledge_agent.ingestion.vector_store import get_client
from aiml_knowledge_agent.models.llm_client import LMStudioClient


# A small but multi-paragraph markdown-ish blob — long enough that the
# SimpleChunker (default chunk_size=512) produces more than one chunk,
# so we exercise batching across multiple points.
SAMPLE_DOC = """
# LangGraph state

LangGraph models the agent as a state machine. The state is a TypedDict
that flows from node to node — each node receives the current state, may
return updates, and the framework merges those updates into the next state.

## Why state machines

Cycles are first-class in LangGraph. A critic node can route back to a
synthesizer node, and the framework tracks how many times you've been
through the loop so you can cap retries. Plain DAG frameworks would have
to either flatten this into one giant node or punt the cycle to the user.

## Persistence

Checkpointers persist state between invocations, which is what enables
human-in-the-loop pauses and conversation memory across calls.
""" * 3


async def test_ingest_pipeline_end_to_end(
   lm_client: LMStudioClient,
   isolated_collection: str,
) -> None:
   """Real chunker → real embedder → real Qdrant. Verify counts and payload shape.

   """

   simple_chunker = SimpleChunker()
   embedder = Embedder(client=lm_client)
   pipeline = IngestionPipeline(simple_chunker, embedder)

   num_of_chunks = await pipeline.ingest_document(
      SAMPLE_DOC,
      metadata={
         "source": "http://localhost:0000/test",
         "doc_type": "markdown",
         "framework": "langgraph",
      },
   )

   assert num_of_chunks > 1

   qdrant_client = get_client()
   # count() returns CountResult — pull the .count out of it.
   assert qdrant_client.count(collection_name=isolated_collection).count == num_of_chunks

   # Pull back every point we just wrote so we can verify all payloads,
   # not just the first few.
   records, _ = qdrant_client.scroll(
      collection_name=isolated_collection,
      limit=num_of_chunks,
      with_payload=True,
   )
   for record in records:
       # User-supplied metadata fields
       assert record.payload["source"] == "http://localhost:0000/test"
       assert record.payload["doc_type"] == "markdown"
       assert record.payload["framework"] == "langgraph"
       # Pipeline-added fields
       assert "text" in record.payload
       assert "chunk_index" in record.payload
       assert "date_ingested" in record.payload



async def test_reingest_replaces_prior_chunks(
    lm_client: LMStudioClient,
    isolated_collection: str,
) -> None:
   """Ingesting the same source twice doesn't accumulate duplicates.
   Inserting SAME source clear the previous chunk.
   """

   simple_chunker = SimpleChunker()
   embedder = Embedder(client=lm_client)
   pipeline = IngestionPipeline(simple_chunker, embedder)

   _ = await pipeline.ingest_document(
      SAMPLE_DOC,
      metadata={
         "source": "http://localhost:0000/test",
         "doc_type": "markdown",
         "framework": "langgraph",
      },
   )

   # Second ingest with the SAME source but DIFFERENT text.
   # Use a separate variable name — assigning to SAMPLE_DOC here would
   # mark it as local for the whole function, making the line above
   # raise UnboundLocalError.
   second_doc = SAMPLE_DOC * 2  # different length → different chunk count
   num_of_chunks = await pipeline.ingest_document(
      second_doc,
      metadata={
         "source": "http://localhost:0000/test",
         "doc_type": "markdown",
         "framework": "langgraph",
      },
   )

   qdrant_client = get_client()
   # If delete_by_source didn't fire, this would be (first_count + num_of_chunks).
   # If it fired correctly, only the second ingest's chunks remain.
   assert qdrant_client.count(collection_name=isolated_collection).count == num_of_chunks
