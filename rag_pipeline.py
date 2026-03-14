import os
import faiss
import numpy as np
import pickle
import google.genai as genai
import json
import zipfile
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_json_col(column):
    def parse_item(x):
        try:
            if pd.isna(x):
                return []
            return json.loads(x.replace("'", '"'))
        except Exception:
            return []
    return column.apply(parse_item)

def extract_names(item_list):
    if isinstance(item_list, list):
        return [item.get("name") for item in item_list if isinstance(item, dict) and "name" in item]
    return []

def get_director(crew_list):
    if isinstance(crew_list, list):
        for member in crew_list:
            if isinstance(member, dict) and member.get("job") == "Director":
                return member.get("name")
    return "Unknown Director"

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

    def auto_setup(self, zip_path="data/archive (2).zip", data_dir="data", output_json="data/processed_movies.json"):
        """Automatically extracts, processes, and ingests vector data if missing."""
        if self.load_index():
            return True # Already setup
            
        print("Vector database not found. Starting automatic setup...")
        os.makedirs(data_dir, exist_ok=True)
        
        movies_csv = os.path.join(data_dir, "tmdb_5000_movies.csv")
        credits_csv = os.path.join(data_dir, "tmdb_5000_credits.csv")
        
        # 1. Extract ZIP
        if not (os.path.exists(movies_csv) and os.path.exists(credits_csv)):
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Missing {zip_path}. Cannot automatically setup the database.")
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                
        # 2. Process Data
        if not os.path.exists(output_json):
            print("Processing CSV data...")
            movies = pd.read_csv(movies_csv)
            credits = pd.read_csv(credits_csv)
            
            if "id" in movies.columns and "movie_id" in credits.columns:
                df = movies.merge(credits, left_on="id", right_on="movie_id")
            else:
                df = movies.merge(credits, on="title")
                
            df["genres"] = parse_json_col(df["genres"])
            df["keywords"] = parse_json_col(df["keywords"])
            df["production_companies"] = parse_json_col(df["production_companies"])
            df["cast"] = parse_json_col(df["cast"])
            df["crew"] = parse_json_col(df["crew"])
            
            df["genres"] = df["genres"].apply(extract_names)
            df["keywords"] = df["keywords"].apply(extract_names)
            df["production_companies"] = df["production_companies"].apply(extract_names)
            df["actors"] = df["cast"].apply(lambda x: [actor.get("name") for actor in x[:3]] if isinstance(x, list) else [])
            df["director"] = df["crew"].apply(get_director)
            
            processed_movies = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
                title = row.get("title") or row.get("original_title") or "Unknown Title"
                release_date = str(row.get("release_date", ""))
                year = release_date.split("-")[0] if "-" in release_date else "Unknown"
                genres = ", ".join(row["genres"]) if row["genres"] else "Unknown"
                actors = ", ".join(row["actors"]) if row["actors"] else "Unknown"
                director = row["director"]
                popularity = row.get("popularity", 0.0)
                overview = row.get("overview") or "No overview available."
                keywords = ", ".join(row["keywords"]) if row["keywords"] else "None"
                companies = ", ".join(row["production_companies"]) if row["production_companies"] else "None"
                
                document = (
                    f"Title: {title}\n"
                    f"Year: {year}\n"
                    f"Genres: {genres}\n"
                    f"Director: {director}\n"
                    f"Actors: {actors}\n"
                    f"Popularity Score: {popularity}\n"
                    f"Overview:\n{overview}\n"
                    f"Keywords:\n{keywords}\n"
                    f"Production Companies:\n{companies}"
                )
                processed_movies.append({
                    "id": int(row.get("id", 0)) if pd.notna(row.get("id")) else 0,
                    "title": title,
                    "document": document,
                    "metadata": {"year": year, "director": director, "popularity": float(popularity)}
                })
                
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(processed_movies, f, indent=4)
                
        # 3. Ingest Data
        self.ingest_data(output_json)
        return True

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
