# Padova Restaurant RAG

> End-to-end Retrieval-Augmented Question Answering over restaurant reviews in Padova.

Padova Restaurant RAG is a deployable AI system that answers natural-language questions about restaurants using restaurant metadata, customer reviews, semantic search, and source-grounded answer generation.

The system is designed as a production-style RAG application: it builds a searchable knowledge base from restaurant reviews, generates dense embeddings, indexes documents with FAISS, retrieves relevant evidence, applies hybrid ranking, and exposes the pipeline through a FastAPI service.

---

## Why This Project

Restaurant reviews contain useful information about food quality, service, price, ambiance, location, and customer experience, but this information is scattered across hundreds or thousands of individual reviews.

This project turns unstructured restaurant reviews into a question-answering system that can answer questions such as:

```text
Where can I find good pizza in Padova?
Which restaurants have the best ambiance?
Which places are good value for money?
Which restaurants have friendly service?
What are the best restaurants for a romantic dinner?
```

---

## Core Features

- Retrieval-Augmented Question Answering over restaurant reviews
- Semantic search with SentenceTransformer embeddings
- FAISS vector indexing for fast similarity search
- Review-level and restaurant-level document construction
- Aspect-aware retrieval for food, service, price, ambiance, and other restaurant features
- Hybrid ranking using semantic similarity, restaurant ratings, review volume, and aspect matching
- Source-grounded answers with retrieved evidence
- FastAPI service for inference
- Dockerized deployment
- CLI support
- Retrieval evaluation
- Automated tests and CI

---

## Architecture

```text
Restaurant Data
      ↓
Data Validation
      ↓
Preprocessing
      ↓
Document Construction
      ↓
Aspect Extraction
      ↓
Embedding Generation
      ↓
FAISS Vector Index
      ↓
Semantic Retrieval
      ↓
Hybrid Ranking
      ↓
Evidence Selection
      ↓
Answer Generation
      ↓
FastAPI / CLI
```

---

## RAG Pipeline

```text
User Question
      ↓
Question Embedding
      ↓
Vector Search
      ↓
Top-k Candidate Documents
      ↓
Hybrid Ranking
      ↓
Evidence Selection
      ↓
Source-Grounded Answer
```

Example response:

```json
{
  "question": "Where can I find good pizza in Padova?",
  "answer": "Based on the retrieved reviews, the strongest options are...",
  "sources": [
    {
      "restaurant": "Example Pizzeria",
      "rating": 4.6,
      "score": 0.84,
      "aspects": ["pizza", "food", "service"],
      "evidence": "Reviewers mention excellent pizza and friendly staff."
    }
  ],
  "metadata": {
    "retrieval_strategy": "hybrid",
    "top_k": 5,
    "latency_ms": 42
  }
}
```

---

## Tech Stack

- Python 3.11
- SentenceTransformers
- FAISS
- pandas
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Docker
- pytest
- GitHub Actions

---

## Repository Structure

```text
padova-restaurant-rag/
│
├── app/                    # FastAPI application
├── src/padova_rag/          # Core RAG package
├── scripts/                 # Index building, CLI, evaluation scripts
├── configs/                 # Configuration files
├── data/                    # Sample and evaluation data
├── indexes/                 # Vector index files
├── tests/                   # Unit and API tests
├── docs/                    # Architecture, API, deployment, evaluation docs
├── .github/workflows/       # CI pipeline
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## API

The FastAPI service exposes:

```text
GET  /health
POST /ask
POST /retrieve
GET  /metadata
```

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where can I find good pizza in Padova?", "top_k": 5}'
```

---

## Run Locally

```bash
git clone https://github.com/rezaalizaadeh/padova-restaurant-rag.git
cd padova-restaurant-rag

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Build the vector index:

```bash
make build-index
```

Run the API:

```bash
make run-api
```

Open the API documentation:

```text
http://localhost:8000/docs
```

---

## Run with Docker

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```

---

## Retrieval and Ranking

The system combines semantic retrieval with metadata-aware ranking.

```text
final_score =
0.65 * semantic_similarity
+ 0.15 * normalized_restaurant_rating
+ 0.10 * normalized_review_count
+ 0.10 * aspect_match_score
```

This improves recommendation-style questions where the best answer depends on both review meaning and restaurant-level quality signals.

---

## Evaluation

The project includes retrieval evaluation using a manually curated evaluation set.

Metrics:

- Recall@k
- MRR
- aspect match rate
- source diversity
- average latency
- answer-has-sources rate

Example result format:

```text
Method                  Recall@5   MRR    Aspect Match   Latency
Dense FAISS              TBD       TBD       TBD          TBD
Hybrid Ranking           TBD       TBD       TBD          TBD
Hybrid + Reranker         TBD       TBD       TBD          TBD
```

---

## Data

The repository includes sample data for reproducibility and testing.

Full restaurant review datasets may not be included because of licensing, privacy, and data usage restrictions. Users can add their own restaurant and review data locally and rebuild the vector index using the provided scripts.

Optional data collection through Google Places API is supported through environment variables.

---

## Roadmap

- [ ] Data validation and preprocessing
- [ ] Review-level and restaurant-level document construction
- [ ] SentenceTransformer embedding pipeline
- [ ] FAISS vector index
- [ ] Dense semantic retrieval
- [ ] Hybrid ranking
- [ ] Source-grounded answer generation
- [ ] FastAPI service
- [ ] Docker deployment
- [ ] Retrieval evaluation
- [ ] Automated tests
- [ ] GitHub Actions CI
- [ ] Optional cross-encoder reranking
- [ ] Optional hosted demo

---

## Project Status

This repository is under active development.

The first milestone is a working local RAG pipeline with FAISS retrieval and source-grounded answers. The next milestones are API serving, Docker deployment, retrieval evaluation, and CI-tested releases.

---

## Author

**Reza Alizaadeh**

Focused on NLP, RAG systems, semantic search, and deployable AI engineering.
