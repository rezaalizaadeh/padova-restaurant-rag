"""
Inference entry point for the NLP exam.

The professor can replace questions.txt and run:

    python exam.py

This file loads the already-created embedding model, saved document embeddings,
FAISS index, and document metadata.  It never rebuilds document embeddings.
"""

import json
import os
import pickle
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from qa_engine import answer_questions


QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


def require_file(path):
    """Raise a clear error when a required submission asset is missing."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {os.path.basename(path)}. "
            "Keep all submitted assets in the same folder as exam.py."
        )


def load_questions(path):
    """Read non-empty numbered or unnumbered questions from a text file."""

    require_file(path)
    with open(path, "r", encoding="utf-8") as file:
        questions = [line.strip() for line in file if line.strip()]

    if not questions:
        raise ValueError("questions.txt does not contain any questions.")

    return questions


def load_saved_system():
    """
    Load and validate all previously built retrieval assets.

    np.load() explicitly loads the saved document-embedding matrix required by
    the assignment.  FAISS performs retrieval over the equivalent vectors in
    faiss.index.
    """

    require_file(CONFIG_FILE)
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    model_path = os.path.join(BASE_DIR, config["model_path"])
    embeddings_path = os.path.join(BASE_DIR, config["embeddings_file"])
    index_path = os.path.join(BASE_DIR, config["faiss_index_file"])
    documents_path = os.path.join(BASE_DIR, config["documents_file"])
    metadata_path = os.path.join(BASE_DIR, config["metadata_file"])

    for path in (
        model_path,
        embeddings_path,
        index_path,
        documents_path,
        metadata_path,
    ):
        require_file(path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Memory mapping avoids copying the entire matrix unnecessarily while
    # still loading and validating the saved embeddings.
    embeddings = np.load(embeddings_path, mmap_mode="r")
    index = faiss.read_index(index_path)

    with open(documents_path, "rb") as file:
        documents = pickle.load(file)

    model = SentenceTransformer(
        model_path,
        device="cpu",
        local_files_only=True,
    )

    expected_documents = int(metadata["number_of_documents"])
    expected_dimension = int(metadata["embedding_dimension"])

    if embeddings.shape != (expected_documents, expected_dimension):
        raise ValueError("embeddings.npy has an unexpected shape.")
    if index.ntotal != expected_documents:
        raise ValueError("faiss.index has an unexpected number of vectors.")
    if index.d != expected_dimension:
        raise ValueError("faiss.index has an unexpected vector dimension.")
    if len(documents) != expected_documents:
        raise ValueError("documents.pkl does not match the saved embeddings.")

    return model, index, documents


def format_question_line(question, position):
    """Preserve existing numbering or add numbering to unnumbered questions."""

    import re

    if re.match(r"^\s*\d+\s*:", question):
        return question.strip()
    return f"{position}: {question.strip()}"


def main():
    """Load the saved RAG system, answer questions, and print to stdout."""

    questions = load_questions(QUESTIONS_FILE)
    model, index, documents = load_saved_system()
    results = answer_questions(questions, model, index, documents)

    for position, result in enumerate(results):
        print(format_question_line(result["question"], position))
        print(result["answer"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
