from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from pinecone import Pinecone
import uvicorn
import traceback  # for better error logging

# Initialize app
app = FastAPI()

# ✅ Initialize Pinecone and Groq clients
pc = Pinecone(api_key="pcsk_3hfQ5V_C4FB4MAqHnSqzLtRpz3CepW8xN8NoKAjuJebJFegZJJGaJ9u8VeerQqQ4kirmfV")
client = Groq(api_key="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")

# ✅ Load embedding models
embedding_models = {
    "bge-small-en-v1.5": SentenceTransformer("BAAI/bge-small-en-v1.5"),
    "LaBSE": SentenceTransformer("sentence-transformers/LaBSE"),
    "distiluse": SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
}

# ✅ Map model names to Pinecone indexes
index_name_map = {
    "bge-small-en-v1.5": "chatbot-bge",
    "LaBSE": "chatbot-labse",
    "distiluse": "chatbot-distiluse"
}

# ✅ Request schema
class QueryRequest(BaseModel):
    question: str
    language: str
    model: str

# ✅ Chat route
@app.post("/chat")
async def chat_with_context(query: QueryRequest):
    try:
        model_name = query.model
        index_name = index_name_map.get(model_name)

        if not index_name:
            return {"error": f"Invalid model name: {model_name}"}

        # Get Pinecone index
        index = pc.Index(index_name)

        # Encode the query
        embed_model = embedding_models[model_name]
        embedded_query = embed_model.encode(query.question).tolist()

        # Search in Pinecone
        response = index.query(vector=embedded_query, top_k=10, include_metadata=True)

        print("\n🔍 Matches from Pinecone:")
        context_chunks = []
        for match in response["matches"]:
            text = match["metadata"].get("text", "")
            if text:
                print(f"Score: {match['score']:.4f}")
                print("Text:", text[:200])
                print("-" * 40)
                context_chunks.append(text)

        # Check if we got any context
        if not context_chunks:
            return {"answer": "Sorry, no relevant info found."}

        # Construct the prompt for Groq
        context = "\n".join(context_chunks)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query.question}\nAnswer:"

        # Call Groq LLM
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        # Print full traceback to terminal
        print("Error occurred:", str(e))
        traceback.print_exc()
        return {"error": "Internal Server Error: " + str(e)}


# ✅ Run the server
if __name__ == "__main__":
    uvicorn.run("rag_chatbot:app", host="127.0.0.1", port=8000, reload=True)

