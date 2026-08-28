"""Simple Harry Potter RAG API exercise.

Task: complete every TODO in this file, then run the API and test /query.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# TASK
# Complete TODO 1, TODO 2, TODO 3, and TODO 4.
# Then run the API and test retrieve, chitchat, and off-topic questions.


# ============================= Setup =============================
load_dotenv()

app = FastAPI(title="Harry Potter RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO 1: Add these values to your .env file and load them here.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K"))

model = SentenceTransformer(EMBEDDING_MODEL) # HERE WE NEED TO LOAD THE SAME EMBEDDING MODEL THAT WE USED TO CREATE THE VECTOR DATABASE, WITH THE SAME DIMENSIONALITY.

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# IF YOU WILL USE ANOTHER LLM FROM ANOTHER PROVIDER, USE THE CORRECT CLASS FROM LANGCHAIN AND PROVIDE THE REQUIRED PARAMETERS.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================

class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================

@app.get("/")
def root():
    return {"name": "Sherlock Holmes RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):


    # TODO 2: Write a prompt that returns only one of these words:
    # retrieve, chitchat, or off-topic.
    ROUTER_SYSTEM_PROMPT = """You classify messages for a Sherlock Holmes book search system.
            Return exactly one label and nothing else:
            retrieve - questions about the books, characters, places, or events
            chitchat - greetings, thanks, or casual conversation, and in this case you can answer the question
            off-topic - anything unrelated to the books, and tell the user that you are only answering questions 
            about the Sherlock Holmes books"""




    route = groq_llm.invoke([
        SystemMessage(content=(ROUTER_SYSTEM_PROMPT)),
        HumanMessage(content=request.query),
    ]).text.strip().lower()

    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"

    if route == "chitchat":

        # TODO 3: Write a short, friendly prompt for chitchat.
        CHITCHAT_SYSTEM_PROMPT = """You are a friendly assistant for a Sherlock Holmes book Q&A system.
            Respond warmly and briefly to greetings, thanks, and casual conversation.
            Keep replies to one or two sentences, and gently invite the user to ask about 
            the Sherlock Holmes stories, characters, or cases."""

        response = groq_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])

        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )

    if route == "off-topic":
        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the Harry Potter books.",
            sources=[],
        )

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    context = "\n\n".join(
        f"Book: {result.payload['book_name']}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Content: {result.payload['content']}"
        for result in results
    )

    # TODO 4: Write a prompt that answers only from the provided context.
    # Tell the model to say "I do not know" when the context is not enough.
    RAG_SYSTEM_PROMPT = """You answer questions about the Sherlock Holmes books using the provided context passages.
                            Rules:
                            - Base your answer on the information in the context, including reasonable inferences from what the passages describe.
                            - Only say "I do not know" if the context contains nothing relevant to the question.
                            - Keep the answer concise and factual.
                            """
    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload["book_name"],
                page_number=result.payload["page_number"],
                score=result.score,
            )
            for result in results
        ],
    )
