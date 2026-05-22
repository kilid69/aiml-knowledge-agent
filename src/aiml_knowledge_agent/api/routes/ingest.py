"""POST /ingest — accept a document and run it through the ingestion pipeline.

The route is the public face of the pipeline. It defines the wire format
(JSON in, JSON out via Pydantic), pulls a long-lived IngestionPipeline
out of app.state (built once by the `lifespan` hook), and runs exactly
one ingest per request.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from aiml_knowledge_agent.ingestion.pipeline import IngestionPipeline


router = APIRouter(tags=["Ingestion"])


# --- request / response models -------------------------------------------


class IngestRequest(BaseModel):
    """Body of POST /ingest."""
    text: str
    source_url: str
    doc_type: str
    framework: str | None = None
    title: str | None = None
    
    # Pydantic note: a field with `= None` is automatically optional;
    # a field without a default is required and missing it → 422.



class IngestResponse(BaseModel):
    """Body of the 200 response from POST /ingest."""

    chunks_ingested: int   # the count returned by pipeline.ingest_document
    source: str            # echo back source_url so the caller can confirm


# --- dependency ----------------------------------------------------------


def get_pipeline(request: Request) -> IngestionPipeline:
    """Resolve the long-lived IngestionPipeline from app.state.

    `lifespan` in main.py builds the pipeline once at startup and stashes
    it on `app.state.pipeline`. FastAPI injects the current `Request` for
    us, which carries a reference to `app`.

    Tests override this function via `app.dependency_overrides[get_pipeline]`
    to inject a `FakePipeline` without touching real services.
    """
    # seam concept
    return request.app.state.pipeline


# --- route ---------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> IngestResponse:
    """Run one document through the ingestion pipeline."""

    metadata = dict(source=body.source_url, 
                    doc_type=body.doc_type)
    
    # we don't want to store None into the payload
    if body.framework:
        metadata.update(framework=body.framework)

    if body.title:
        metadata.update(title=body.title)
    
    chunk_count = await pipeline.ingest_document(body.text, metadata=metadata)
    
    return IngestResponse(chunks_ingested=chunk_count, source=body.source_url)
