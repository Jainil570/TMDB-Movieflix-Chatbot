# 🎬 Movie Knowledge Chatbot
DEMO : https://screenrec.com/share/WMJqmn9LFE
This is a RAG (Retrieval Augmented Generation) based chatbot built using the **TMDB 5000 Movie Dataset**. It answers user queries about movies relying *strictly* on the provided dataset. 

This project uses:
- **Google Gemini API** (`gemini-2.5-flash` for text generation)
- **Sentence-Transformers** (`all-MiniLM-L6-v2` for local text embeddings)
- **FAISS** (Local vector database)
- **Streamlit** (Web App framework)
- **Pandas** (Data processing)

---

## 🚀 Setup & Execution (Cloud/Local)

To run this pipeline and chatbot on a cloud environment or your local machine, follow these steps:

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
1. Create a `.env` file from the provided `.env.example`.
2. Add your Google Gemini API key:
   ```env
   GEMINI_API_KEY="your_actual_api_key_here"
   ```

### 3. Add the Dataset
Create a `data/` folder in the root directory and place your TMDB 5000 CSV files inside:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

### 4. Process the Data
Run the data processing script. This cleans the dataset, parses JSON strings, and formats the data for embedding.
```bash
python process_data.py
```
*This will generate `data/processed_movies.json`.*

### 5. Ingest into Vector Database
Run the pipeline script with the `--ingest` flag. This will convert the processed movies into text embeddings and store them in a FAISS index.
```bash
python rag_pipeline.py --ingest
```
*This will create the `vector_store/` directory.*

### 6. Run the Chatbot Web App
Start the Streamlit application to chat with the system:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

- `process_data.py`: Data cleaning and formatting pipeline.
- `rag_pipeline.py`: FAISS vector DB indexer and LLM generation logic.
- `app.py`: Streamlit chat interface.
- `requirements.txt`: Python package dependencies.
- `.env.example`: Template for the environment variables.
