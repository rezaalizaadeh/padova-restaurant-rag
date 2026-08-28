# Question Answering with RAG for Padova Restaurants

## Overview

This project implements a local Retrieval-Augmented Question Answering system
about restaurants in Padova. It searches restaurant metadata and customer
reviews, ranks relevant evidence, and produces grounded natural-language
answers. The inference program works without an API key or internet access.

The project was developed for the NLP Course A.A. 2025/2026 and is compatible
with Python 3.11.

## Dataset

Following the instructor's recommendation, the submitted model and official
answers use the instructor-provided dataset. This makes the system directly
comparable with the expected course results. The top-level input files contain:

- `places.csv`: 513 restaurants and 21 metadata columns;
- `reviews.csv`: 2,565 raw review rows and 5 columns;
- 2,557 non-empty, deduplicated reviews after preprocessing;
- `questions.txt`: 23 official evaluation questions.

The two CSV files are joined using `place_id`. All 513 identifiers occur in
both files. Preprocessing removes two empty reviews and six repeated
restaurant-review pairs.

For the optional data-collection bonus, a second dataset was collected
independently on 26 August 2026 with Google Places API (New). The collection
found 518 unique identifiers, downloaded 502 restaurants and 2,433 review
texts, and produced a strict Padova subset of 451 restaurants and 2,176 clean
reviews. The collection scripts, raw exports, strict Padova files, state files,
and provenance report are included under `data_collection/`. They are evidence
of independent collection but are deliberately not used to build the submitted
model, in accordance with the instructor's recommendation.

Google Places returns a small sample of reviews for each restaurant (at most
five in this collection). The independent dataset overlaps the instructor data
on 174 restaurant identifiers. The API key is entered invisibly at runtime,
is never written to disk, and is not included in the submission. `exam.py` is
fully offline and never calls Google APIs.

## RAG pipeline

### 1. Preprocessing

`full-code.py`:

1. loads and validates the input files;
2. joins places and reviews through `place_id`;
3. removes empty and duplicate reviews;
4. normalizes text, place types, opening hours, numeric fields, and boolean
   metadata.

### 2. Document construction

The knowledge base contains two complementary document types:

- 513 `restaurant_facts` documents with ratings, cuisine/type, address, price
  level, delivery, dine-in, reservations, meals, drinks, and opening hours;
- 2,557 `review` documents containing restaurant identity and customer-review
  evidence.

The final knowledge base therefore contains 3,070 documents.

### 3. Embeddings and vector index

The pretrained `sentence-transformers/all-MiniLM-L6-v2` bi-encoder converts
each document and question into a 384-dimensional dense vector. Embeddings are
L2-normalized.

The project uses `faiss.IndexFlatIP`. Because vectors are normalized, inner
product is equivalent to cosine similarity. Exact vector search is appropriate
for this small collection and avoids approximate-search loss.

The model and document vectors are generated only by `full-code.py` and saved
locally. `exam.py` loads them and does not recreate document embeddings.

### 4. Query-aware retrieval

Dense similarity is used as the retrieval baseline, then query-aware logic:

- recognizes exact restaurant names;
- distinguishes factual questions from review-based questions;
- reads delivery, dine-in, price, and opening-hours values from structured
  restaurant metadata;
- detects cuisine and aspects such as ambiance, service, vegan, gluten-free,
  halal, affordability, family suitability, outdoor seating, and live music;
- aggregates multiple documents by `place_id` so each restaurant appears once;
- combines semantic similarity, rating quality, rating confidence, popularity,
  explicit aspect evidence, and cuisine/type matches.

A Bayesian-adjusted rating reduces the chance that a restaurant with very few
ratings outranks a well-supported restaurant solely because of a perfect raw
score.

### 5. Grounded answer generation

The answer generator is deterministic and evidence-based:

- factual questions use structured fields;
- recommendation questions return unique ranked restaurants with ratings and
  review excerpts;
- unsupported requests are reported honestly.

For example, the dataset has no official hygiene-rating, outdoor-seating, or
live-music column. For those questions, the system explicitly says that it is
using review mentions rather than presenting them as verified structured
facts.

## Files

- `exam.py`: loads saved assets, reads `questions.txt`, and prints answers;
- `full-code.py`: reproduces preprocessing, documents, embeddings, FAISS index,
  retrieval assets, and official answers from scratch;
- `qa_engine.py`: shared query analysis, reranking, and answer-generation logic;
- `evaluate.py`: reproducible integrity and behavior checks;
- `data_collection/`: independent Google collection scripts, capped request
  state, raw exports, strict Padova data, backups of the instructor data, and a
  machine-readable provenance report;
- `questions.txt`: questions read at runtime;
- `Question-Answer.txt`: answers for the instructor's official questions;
- `places.csv`, `reviews.csv`: source data;
- `embedding_model/`: local pretrained embedding model;
- `embeddings.npy`: saved normalized document embeddings;
- `faiss.index`: saved FAISS search index;
- `documents.pkl`: document text and structured evidence in vector order;
- `metadata.json`, `config.json`: reproducibility and asset configuration;
- `requirements.txt`: required Python packages.

## Installation

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv` directory is for local development and is not included in the
submission ZIP.

## Run the exam program

Keep all submitted files in the same folder and run:

```bash
python exam.py
```

The program reads every non-empty line in `questions.txt`. It supports both:

```text
0: What are the best restaurants in the city?
```

and:

```text
What are the best restaurants in the city?
```

It prints each question followed by its answer to standard output.

## Validation

Run the included checks with:

```bash
python evaluate.py
```

The checks cover exact restaurant-name routing, factual delivery and dine-in
answers, opening-hours grounding, aspect evidence, restaurant deduplication,
and explicit keto/low-carbohydrate evidence grounding.

## Reproduce the complete project

To rebuild the knowledge base, saved embeddings, FAISS index, and
`Question-Answer.txt`:

```bash
python full-code.py
```

If `embedding_model/` is already present, the local model is used. If it is
absent, `full-code.py` downloads the named pretrained model and saves it for
later offline inference.

## Limitations

- The model dataset was supplied by the instructor and was downloaded more
  than one year before the course assignment; facts may therefore be outdated.
- Google Places supplies at most five review samples per restaurant in the
  separately included bonus collection.
- Review statements can be subjective or outdated.
- The source data has no official hygiene score and no structured fields for
  outdoor seating or live music.
- Questions containing literal placeholders such as `Landmark` or
  `Neighborhood` cannot support a verified proximity answer until a real
  location is supplied.
- The system uses deterministic evidence-based generation rather than an
  external generative API, ensuring reproducible offline execution.
