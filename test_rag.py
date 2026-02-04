import os
import torch
from sentence_transformers import SentenceTransformer
from rag.index import load_embeddings_and_docs, build_and_save_embeddings
from rag.util import answerWithRAG
from pathlib import Path

def test_rag(query):
    """
    Text RAG ability.
    """
    print("=" * 60)
    print("[TEST]  Starting RAG Functionality Test")
    print("=" * 60)

    if not os.path.exists("rag_data"):
        os.makedirs("rag_data", exist_ok=True)

    emb_path = os.path.join("rag_data", "embeddings.npy")
    docs_path = os.path.join("rag_data", "docs.pkl")

    if not os.path.exists(emb_path) or not os.path.exists(docs_path):
        print("[INFO] Embeddings not found, building new ones...")
        build_and_save_embeddings()
    else:
        print("[INFO] Embeddings found, skipping build step.")

    print("[INFO] Loading embeddings and documents...")
    embeddings_np, docs, docsDict = load_embeddings_and_docs()
    embeddings = torch.tensor(embeddings_np)

    print("[INFO] Loading embedding model...")
    model_path = Path(__file__).parent / "models" / "mxbai-embed-large-v1"

    model_path = model_path.resolve()

    print("[DEBUG] Using absolute model path:", model_path)

    embeddingModel = SentenceTransformer(str(model_path))

    print("=" * 60)
    print(f"[QUERY] {query}")
    print("=" * 60)
    answer = answerWithRAG(query, embeddings, embeddingModel, docs, docsDict)

    print("\n[ANSWER CONTEXT]")
    print("-" * 60)
    print(answer if answer else " No relevant context found.")
    print("-" * 60)

    print("[TEST]  RAG test completed successfully.")


if __name__ == "__main__":
    test_rag("Please analyze the data and provide insights on:1. Key patterns in successful vs unsuccessful runs.2. Parameter ranges that appear promising.3. Any timing or wirelength trends.4. Recommendations for subsequent runs.")
