import re
import glob
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from llama_index.core.readers import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from pydantic import BaseModel

# === API Keys and Config ===
GROQ_API_KEY = "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"
PINECONE_API_KEY = "pcsk_3hfQ5V_C4FB4MAqHnSqzLtRpz3CepW8xN8NoKAjuJebJFegZJJGaJ9u8VeerQqQ4kirmfV"
PINECONE_REGION = "us-east-1"

# === Initialize Clients ===
pinecone = Pinecone(api_key=PINECONE_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

# === FastAPI App ===
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)  # No Content

# === Text Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Chunking Strategies ===
def fixed_chunk(text, max_tokens=512):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def recursive_character_chunk(text, max_chars=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

# === Core Vector Generation and Storage ===
def generate_and_store_vectors():
    try:
        embedding_models = {
            "bge-small-en-v1.5": {
                "model": SentenceTransformer("BAAI/bge-small-en-v1.5"),
                "dimension": 384,
                "index_name": "chatbot-bge"
            },
            "LaBSE": {
                "model": SentenceTransformer("sentence-transformers/LaBSE"),
                "dimension": 768,
                "index_name": "chatbot-labse"
            },
            "distiluse": {
                "model": SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2"),
                "dimension": 512,
                "index_name": "chatbot-distiluse"
            }
        }

        strategies = {
            "fixed": fixed_chunk,
            "recursive": recursive_character_chunk
        }

        language_configs = {
            "english": [file for file in glob.glob("*.pdf") if not file.lower().startswith("es")],
            "spanish": [file for file in glob.glob("es*.pdf")]
        }

        total_vectors = 0

        for language, pdf_files in language_configs.items():
            print(f"\nProcessing {language.upper()} files: {pdf_files}")

            for file in pdf_files:
                print(f"\nReading file: {file}")
                documents = SimpleDirectoryReader(input_files=[file]).load_data()
                full_text = "\n".join([doc.text for doc in documents])
                clean_text = preprocess_text(full_text)

                for strategy_name, chunk_func in strategies.items():
                    chunks = chunk_func(clean_text)
                    print(f"\uD83D\uDD39 {strategy_name} chunking → {len(chunks)} chunks")

                    for model_name, config in embedding_models.items():
                        model = config["model"]
                        dimension = config["dimension"]
                        index_name = config["index_name"]

                        if index_name not in [index.name for index in pinecone.list_indexes()]:
                            pinecone.create_index(
                                name=index_name,
                                dimension=dimension,
                                metric="cosine",
                                spec={"pod_type": "p1"}
                            )
                            print(f"Created Pinecone index: {index_name}")

                        index = pinecone.Index(index_name)
                        embeddings = model.encode(chunks, show_progress_bar=True)

                        vectors = [{
                            "id": f"{file}-{strategy_name}-{model_name}-{i}",
                            "values": emb.tolist(),
                            "metadata": {
                                "file": file,
                                "chunk_id": i,
                                "strategy": strategy_name,
                                "model": model_name,
                                "language": language,
                                "text": chunks[i]
                            }
                        } for i, emb in enumerate(embeddings)]

                        index.upsert(vectors=vectors)
                        print(f"Uploaded {len(vectors)} vectors to index: {index_name}")
                        total_vectors += len(vectors)

        return {"message": "All vectors inserted successfully.", "total_vectors": total_vectors}

    except Exception as e:
        print(f"Exception in vector generation/storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === FastAPI Endpoint ===
@app.post("/insert")
async def insert_data():
    try:
        print("Existing indexes:", [index.name for index in pinecone.list_indexes()])
        result = generate_and_store_vectors()
        return result
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Query Request Model ===
class QueryRequest(BaseModel):
    query: str
    model: str
    top_k: int = 5

# === Query Endpoint ===
@app.post("/query")
async def query_pinecone(request: QueryRequest):
    try:
        model_config = {
            "bge-small-en-v1.5": {
                "model": SentenceTransformer("BAAI/bge-small-en-v1.5"),
                "index": "chatbot-bge"
            },
            "LaBSE": {
                "model": SentenceTransformer("sentence-transformers/LaBSE"),
                "index": "chatbot-labse"
            },
            "distiluse": {
                "model": SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2"),
                "index": "chatbot-distiluse"
            }
        }

        if request.model not in model_config:
            raise HTTPException(status_code=400, detail="Invalid model name.")

        model = model_config[request.model]["model"]
        index_name = model_config[request.model]["index"]
        index = pinecone.Index(index_name)

        vector = model.encode(request.query).tolist()

        result = index.query(
            vector=vector,
            top_k=request.top_k,
            include_metadata=True
        )

        return {
            "query": request.query,
            "results": [
                {
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "file": match["metadata"].get("file", ""),
                    "chunk_id": match["metadata"].get("chunk_id", "")
                }
                for match in result.get("matches", [])
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
