import os
import faiss
import numpy as np
import pickle
import google.genai as genai
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class MovieRAG:
    def __init__(self, index_dir="vector_store"):
        self.index_dir = index_dir

        # Initialize Google GenAI client for generation model
        # The genai.Client() will automatically pick up the configured API key
        # if genai.configure() has been called previously.
        self.genai_client = genai.Client()
        self.generation_model = "gemini-2.5-flash" # Gemini for NLP

        # Initialize Sentence Transformer for embeddings
        self.embedding_model_name = "all-MiniLM-L6-v2"
        print(f"Loading SentenceTransformer model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.index = None
        self.documents = []

        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "movie_index.faiss")
        self.docs_path = os.path.join(self.index_dir, "movie_docs.pkl")

    def get_embedding(self, text):
        # Use SentenceTransformer for embeddings
        # SentenceTransformer returns numpy array, convert to list for consistency
        return self.embedding_model.encode(text).tolist()

    def ingest_data(self, json_path):
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} movies. Generating embeddings...")
        self.documents = data
        embeddings = []

        # We can adjust this limit depending on resources, removing for full ingestion if needed,
        # but the provided notebook limited this to 500 for demonstration. Leaving as is for safety.
        subset = self.documents[:500]

        for i, item in enumerate(tqdm(subset, desc="Embedding")):
            try:
                emb = self.get_embedding(item['document'])
                embeddings.append(emb)
            except Exception as e:
                print(f"Error generating embedding for item {i}: {e}. Skipping item.")

        self.documents = subset

        embedding_matrix = np.array(embeddings).astype('float32')
        dimension = embedding_matrix.shape[1]

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embedding_matrix)

        self.save_index()
        print("Ingestion complete.")

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Index saved to {self.index_path}")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            return True
        return False

    def retrieve(self, query, top_k=5):
        if self.index is None:
            if not self.load_index():
                raise FileNotFoundError("Vector index not found. Please ingest data first.")

        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def generate_answer(self, query):
        retrieved_docs = self.retrieve(query, top_k=5)
        if not retrieved_docs:
             return "I couldn't find any relevant information in the movie database."

        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"--- Movie {i+1} ---\n{doc['document']}\n\n"

        system_prompt = (
            "You are an expert movie chatbot. You must answer the user's question "
            "based ONLY on the provided movie dataset context below. "
            "If the answer cannot be found in the context, do not guess or hallucinate. "
            "Instead, politely state that you can only answer based on the provided dataset and you do not have that information.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )

        response = self.genai_client.models.generate_content(
            model=self.generation_model,
            contents=system_prompt
        )
        return response.text

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configure API key for standalone execution if needed
    if not os.getenv("GEMINI_API_KEY"):
         print("Warning: GEMINI_API_KEY environment variable not set.")

    parser = argparse.ArgumentParser(description="Movie RAG Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run the ingestion process")
    parser.add_argument("--data", type=str, default="data/processed_movies.json", help="Path to processed JSON data")
    
    args = parser.parse_args()
    
    rag = MovieRAG(index_dir="vector_store")
    
    if args.ingest:
        if os.path.exists(args.data):
            rag.ingest_data(args.data)
        else:
            print(f"Cannot find {args.data}. Data processing step failed or path is incorrect.")
    else:
        print("Use --ingest to ingest data, or import MovieRAG in another script to use the pipeline.")
