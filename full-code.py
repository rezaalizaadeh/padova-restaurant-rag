import ast
import json
import os
import pickle
import re

# Store temporary Hugging Face downloads inside the project instead of relying
# on a computer-specific global cache directory.
PROJECT_MODEL_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".model_cache",
)
os.environ.setdefault("HF_HOME", PROJECT_MODEL_CACHE)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qa_engine import answer_questions, write_question_answer_file


# ============================================================================
# 1. File paths
# ============================================================================

# Get the folder where this Python file is located.
# This makes the code portable: it works on your laptop and professor's laptop.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create paths to the input files.
# These files must be in the same folder as full-code.py.
PLACES_FILE = os.path.join(BASE_DIR, "places.csv")
REVIEWS_FILE = os.path.join(BASE_DIR, "reviews.csv")
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
QUESTION_ANSWER_FILE = os.path.join(BASE_DIR, "Question-Answer.txt")

# Pretrained ML model used to convert text into semantic vectors.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Files created by full-code.py and later loaded by exam.py.
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "embedding_model")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss.index")
DOCUMENTS_FILE = os.path.join(BASE_DIR, "documents.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


# ============================================================================
# 2. Load data
# ============================================================================

def load_data():
    """
    Load the input files:
    - places.csv
    - reviews.csv
    - questions.txt
    """

    # Read places.csv as a pandas DataFrame.
    places_df = pd.read_csv(PLACES_FILE)

    # Read reviews.csv as a pandas DataFrame.
    reviews_df = pd.read_csv(REVIEWS_FILE)

    # Read questions.txt line by line.
    # "as f" means: call the opened file object f.
    # f is just a short variable name for "file".
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    return places_df, reviews_df, questions


# ============================================================================
# 3. Check place_id connection
# ============================================================================

def check_place_id_connection(places_df, reviews_df):
    """
    Check whether places.csv and reviews.csv can be connected using place_id.

    places.csv has restaurant metadata.
    reviews.csv has customer reviews.
    place_id is the bridge between them.
    """

    # Get all unique place_id values from places.csv.
    # dropna() removes missing IDs.
    # astype(str) converts IDs to text, so matching is safer.
    # set(...) keeps only unique values.
    places_ids = set(places_df["place_id"].dropna().astype(str))

    # Get all unique place_id values from reviews.csv.
    review_ids = set(reviews_df["place_id"].dropna().astype(str))

    # IDs that exist in both files.
    common_ids = places_ids.intersection(review_ids)

    # IDs that appear in reviews.csv but not in places.csv.
    reviews_without_place = review_ids - places_ids

    # IDs that appear in places.csv but not in reviews.csv.
    places_without_reviews = places_ids - review_ids

    print("\nPLACE_ID CONNECTION CHECK")
    print("-" * 80)
    print(f"Unique place_ids in places.csv: {len(places_ids)}")
    print(f"Unique place_ids in reviews.csv: {len(review_ids)}")
    print(f"Common place_ids: {len(common_ids)}")
    print(f"Place IDs in reviews but not in places: {len(reviews_without_place)}")
    print(f"Place IDs in places but not in reviews: {len(places_without_reviews)}")


# ============================================================================
# 4. Merge places and reviews
# ============================================================================

def merge_places_and_reviews(places_df, reviews_df):
    """
    Merge places and reviews into one table.

    Each review receives the corresponding restaurant metadata.
    We use a left join from reviews to places:
    - keep every review
    - add matching place information
    """

    merged_df = reviews_df.merge(
        places_df,        # right table: restaurant metadata
        on="place_id",    # common column used for matching
        how="left"        # keep all rows from reviews_df
    )

    return merged_df


# ============================================================================
# 5. Inspect merged data
# ============================================================================

def inspect_merged_data(merged_df):
    """
    Inspect the merged table after combining places and reviews.
    This does not change the data. It only prints information.
    """

    print("\nMERGED DATA INSPECTION")
    print("-" * 80)

    # Shape means: number of rows and columns.
    print(f"Merged shape: {merged_df.shape}")

    # Print all column names.
    print("\nMerged columns:")
    for col in merged_df.columns:
        print(f" - {col}")

    # Show only useful columns, not all 25 columns.
    print("\nFirst 3 merged rows:")
    columns_to_show = [
        "place_id",
        "place_name",
        "place_average_ratings",
        "place_ratings_count",
        "place_primary_type",
        "review_rating",
        "review",
    ]

    # Keep only columns that really exist in merged_df.
    # This prevents errors if a column name is missing or changed.
    existing_columns = [col for col in columns_to_show if col in merged_df.columns]

    # Print first 3 rows using only selected columns.
    print(merged_df[existing_columns].head(3))

    # Count missing restaurant names.
    print("\nMissing restaurant names after merge:")
    print(merged_df["place_name"].isna().sum())

    # Count missing review texts.
    print("\nMissing review texts:")
    print(merged_df["review"].isna().sum())


# ============================================================================
# 6. Clean text values
# ============================================================================

def clean_text(value):
    """
    Clean one text value.

    This function is used for:
    - reviews
    - restaurant names
    - addresses
    - restaurant types
    - opening hours
    """

    # If the value is missing, return an empty string.
    if pd.isna(value):
        return ""

    # Convert the value to string, because text operations work on strings.
    text = str(value)

    # Replace line breaks with spaces.
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Remove repeated spaces.
    # Example: "good     food" becomes "good food".
    text = " ".join(text.split())

    # Remove spaces from beginning and end.
    return text.strip()


def parse_list_field(value):
    """
    Convert a CSV value containing a Python-style list into a real list.

    Example input:
    "['indian_restaurant', 'pizza_restaurant']"

    Example output:
    ['indian_restaurant', 'pizza_restaurant']

    ast.literal_eval() reads basic Python literals safely. Unlike eval(), it
    does not execute arbitrary Python code.
    """

    if pd.isna(value):
        return []

    text = clean_text(value)

    if not text:
        return []

    try:
        parsed_value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # If a value is not a valid serialized list, preserve it as one item.
        return [text]

    if isinstance(parsed_value, list):
        return [clean_text(item) for item in parsed_value if clean_text(item)]

    return [clean_text(parsed_value)]


def clean_place_types(value):
    """
    Convert serialized place types into natural searchable text.

    Example:
    "['indian_restaurant', 'pizza_restaurant']"
    becomes:
    "indian restaurant, pizza restaurant"
    """

    place_types = parse_list_field(value)
    readable_types = [place_type.replace("_", " ") for place_type in place_types]

    if not readable_types:
        return "Unknown"

    return ", ".join(readable_types)


def clean_opening_hours(value):
    """
    Convert serialized opening hours into readable text.

    The source CSV contains literal Unicode escape sequences such as
    "\\u2009" and "\\u202f". These represent narrow spaces, so we replace
    them with ordinary spaces and join the weekday entries with semicolons.
    """

    opening_hours = parse_list_field(value)

    if not opening_hours:
        return "Unknown"

    cleaned_hours = []

    for entry in opening_hours:
        entry = entry.replace("\\u2009", " ")
        entry = entry.replace("\\u202f", " ")
        entry = entry.replace("\\u00a0", " ")
        entry = entry.replace("\\u2013", "–")
        entry = " ".join(entry.split())
        cleaned_hours.append(entry)

    return "; ".join(cleaned_hours)


# ============================================================================
# 7. Clean merged data
# ============================================================================

def clean_data(merged_df):
    """
    Clean the merged restaurant-review dataset.

    This prepares the data for the next step: building RAG documents.
    """

    print("\nCLEANING DATA")
    print("-" * 80)

    # Count rows before cleaning.
    original_rows = len(merged_df)
    print(f"Rows before cleaning: {original_rows}")

    # ------------------------------------------------------------------------
    # 7.1 Clean review text
    # ------------------------------------------------------------------------

    # Apply clean_text() to every value in the review column.
    merged_df["review"] = merged_df["review"].apply(clean_text)

    # Remove rows where review text is empty.
    # Empty reviews are useless for RAG.
    merged_df = merged_df[merged_df["review"] != ""].copy()

    rows_after_empty_review_removal = len(merged_df)
    print(f"Rows after removing empty reviews: {rows_after_empty_review_removal}")
    print(f"Removed empty reviews: {original_rows - rows_after_empty_review_removal}")

    # ------------------------------------------------------------------------
    # 7.2 Clean important text columns
    # ------------------------------------------------------------------------

    text_columns = [
        "place_name",
        "place_address",
        "place_primary_type",
        "place_price_level",
    ]

    for col in text_columns:
        if col in merged_df.columns:
            # Clean spaces and line breaks.
            merged_df[col] = merged_df[col].apply(clean_text)

            # Replace empty text with "Unknown".
            merged_df[col] = merged_df[col].replace("", "Unknown")

    # These two columns contain serialized lists and require specialized
    # cleaning rather than ordinary whitespace cleaning.
    merged_df["place_types"] = merged_df["place_types"].apply(clean_place_types)
    merged_df["places_opening_hours"] = merged_df[
        "places_opening_hours"
    ].apply(clean_opening_hours)

    # ------------------------------------------------------------------------
    # 7.3 Convert rating/count columns to numbers
    # ------------------------------------------------------------------------

    numeric_columns = [
        "place_average_ratings",
        "place_ratings_count",
        "review_rating",
    ]

    for col in numeric_columns:
        if col in merged_df.columns:
            # Convert to numeric.
            # If conversion fails, errors="coerce" changes the value to NaN.
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

            # Replace missing numeric values with 0.
            # Here 0 means unknown/missing, not a real rating of zero.
            merged_df[col] = merged_df[col].fillna(0)

    # ------------------------------------------------------------------------
    # 7.4 Remove duplicate reviews
    # ------------------------------------------------------------------------

    before_duplicates = len(merged_df)

    # A duplicate is same restaurant ID + same review text.
    merged_df = merged_df.drop_duplicates(subset=["place_id", "review"]).copy()

    after_duplicates = len(merged_df)

    print(f"Rows after removing duplicate reviews: {after_duplicates}")
    print(f"Removed duplicate reviews: {before_duplicates - after_duplicates}")

    # Count how many cleaned reviews from our submitted dataset are available
    # for each restaurant. This is different from Google's total rating count.
    merged_df["dataset_review_count"] = (
        merged_df.groupby("place_id")["review"].transform("count")
    )

    # ------------------------------------------------------------------------
    # 7.5 Final cleaning summary
    # ------------------------------------------------------------------------

    print(f"Final cleaned rows: {len(merged_df)}")

    print("\nExample cleaned row:")
    columns_to_show = [
        "place_name",
        "place_average_ratings",
        "place_ratings_count",
        "place_primary_type",
        "review_rating",
        "review",
    ]

    existing_columns = [col for col in columns_to_show if col in merged_df.columns]
    print(merged_df[existing_columns].head(1))

    return merged_df


# ============================================================================
# 8. Format values for readable RAG documents
# ============================================================================

def format_boolean(value):
    """
    Convert a boolean-like CSV value into readable text.

    Examples:
    True or "true"  -> "Yes"
    False or "false" -> "No"
    Missing value    -> "Unknown"
    """

    # Missing information should not be described as Yes or No.
    if pd.isna(value):
        return "Unknown"

    # Convert the value to lowercase text so values such as True, "TRUE",
    # and "true" are handled in the same way.
    normalized_value = str(value).strip().lower()

    if normalized_value in {"true", "1", "yes"}:
        return "Yes"

    if normalized_value in {"false", "0", "no"}:
        return "No"

    return "Unknown"


def format_number(value):
    """
    Format numeric values without unnecessary decimal zeros.

    Examples:
    234.0 -> "234"
    4.9   -> "4.9"
    """

    numeric_value = float(value)

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return str(numeric_value)


# ============================================================================
# 9. Build RAG documents
# ============================================================================

def build_rag_documents(cleaned_df):
    """
    Convert the cleaned table into documents that can later be embedded.

    We create two document types:

    1. Restaurant-fact documents:
       One document per restaurant, containing factual metadata such as
       cuisine, rating, delivery, dine-in, and opening hours.

    2. Review documents:
       One document per cleaned review, containing the review text and the
       restaurant identity.

    Each document is stored as a dictionary. The "text" field will be passed
    to the embedding model. The other fields will later help us rank results
    and produce precise answers.
    """

    print("\nBUILDING RAG DOCUMENTS")
    print("-" * 80)

    # This empty list will eventually contain every RAG document.
    documents = []

    # cleaned_df contains one row per review, so restaurant metadata is
    # repeated. Keeping the first row for each place_id gives us exactly
    # one metadata row per restaurant.
    unique_places_df = cleaned_df.drop_duplicates(subset=["place_id"])

    # ------------------------------------------------------------------------
    # 9.1 Create one factual document for every restaurant
    # ------------------------------------------------------------------------

    # iterrows() returns two things for every row:
    # - the row index, which we do not need and therefore name "_"
    # - the actual row data, which we name "row"
    for _, row in unique_places_df.iterrows():

        # An f-string allows real column values to be inserted inside text
        # using expressions such as {row["place_name"]}.
        fact_text = (
            f"Restaurant: {row['place_name']}. "
            f"Primary type: {row['place_primary_type']}. "
            f"Place types: {row['place_types']}. "
            f"Address: {row['place_address']}. "
            f"Average rating: {format_number(row['place_average_ratings'])}. "
            f"Number of ratings: {format_number(row['place_ratings_count'])}. "
            f"Reviews available in this dataset: "
            f"{format_number(row['dataset_review_count'])}. "
            f"Price level: {row['place_price_level']}. "
            f"Delivery: {format_boolean(row['has_delivery'])}. "
            f"Dine-in: {format_boolean(row['has_dine_in'])}. "
            f"Reservable: {format_boolean(row['is_reservable'])}. "
            f"Serves breakfast: {format_boolean(row['serves_breakfast'])}. "
            f"Serves lunch: {format_boolean(row['serves_lunch'])}. "
            f"Serves dinner: {format_boolean(row['serves_dinner'])}. "
            f"Serves beer: {format_boolean(row['serves_beer'])}. "
            f"Serves wine: {format_boolean(row['serves_wine'])}. "
            f"Opening hours: {row['places_opening_hours']}."
        )

        # append() adds one new dictionary to the documents list.
        documents.append(
            {
                "document_type": "restaurant_facts",
                "place_id": str(row["place_id"]),
                "place_name": row["place_name"],
                "place_average_ratings": float(row["place_average_ratings"]),
                "place_ratings_count": int(row["place_ratings_count"]),
                "dataset_review_count": int(row["dataset_review_count"]),
                "place_primary_type": row["place_primary_type"],
                "place_types": row["place_types"],
                "place_address": row["place_address"],
                "place_price_level": row["place_price_level"],
                "has_delivery": format_boolean(row["has_delivery"]),
                "has_dine_in": format_boolean(row["has_dine_in"]),
                "is_reservable": format_boolean(row["is_reservable"]),
                "serves_breakfast": format_boolean(row["serves_breakfast"]),
                "serves_lunch": format_boolean(row["serves_lunch"]),
                "serves_dinner": format_boolean(row["serves_dinner"]),
                "serves_beer": format_boolean(row["serves_beer"]),
                "serves_wine": format_boolean(row["serves_wine"]),
                "places_opening_hours": row["places_opening_hours"],
                "review_rating": None,
                "review_text": None,
                "text": fact_text,
            }
        )

    # ------------------------------------------------------------------------
    # 9.2 Create one semantic document for every cleaned review
    # ------------------------------------------------------------------------

    for _, row in cleaned_df.iterrows():
        review_text = (
            f"Restaurant: {row['place_name']}. "
            f"Primary type: {row['place_primary_type']}. "
            f"Place types: {row['place_types']}. "
            f"Address: {row['place_address']}. "
            f"Restaurant average rating: "
            f"{format_number(row['place_average_ratings'])}. "
            f"Review rating: {format_number(row['review_rating'])}. "
            f"Customer review: {row['review']}"
        )

        documents.append(
            {
                "document_type": "review",
                "place_id": str(row["place_id"]),
                "place_name": row["place_name"],
                "place_average_ratings": float(row["place_average_ratings"]),
                "place_ratings_count": int(row["place_ratings_count"]),
                "dataset_review_count": int(row["dataset_review_count"]),
                "place_primary_type": row["place_primary_type"],
                "place_types": row["place_types"],
                "place_address": row["place_address"],
                "place_price_level": row["place_price_level"],
                "has_delivery": format_boolean(row["has_delivery"]),
                "has_dine_in": format_boolean(row["has_dine_in"]),
                "is_reservable": format_boolean(row["is_reservable"]),
                "serves_breakfast": format_boolean(row["serves_breakfast"]),
                "serves_lunch": format_boolean(row["serves_lunch"]),
                "serves_dinner": format_boolean(row["serves_dinner"]),
                "serves_beer": format_boolean(row["serves_beer"]),
                "serves_wine": format_boolean(row["serves_wine"]),
                "places_opening_hours": row["places_opening_hours"],
                "review_rating": float(row["review_rating"]),
                "review_text": row["review"],
                "text": review_text,
            }
        )

    # Count the two document types separately.
    # In Python, True behaves like 1 and False behaves like 0 inside sum().
    fact_count = sum(
        document["document_type"] == "restaurant_facts"
        for document in documents
    )

    review_count = sum(
        document["document_type"] == "review"
        for document in documents
    )

    print(f"Restaurant-fact documents: {fact_count}")
    print(f"Review documents: {review_count}")
    print(f"Total RAG documents: {len(documents)}")

    # Print one example of each type so we can verify the generated text.
    print("\nExample restaurant-fact document:")
    print(documents[0]["text"])

    print("\nExample review document:")
    print(documents[fact_count]["text"])

    return documents


# ============================================================================
# 10. Create embeddings and save the retrieval system
# ============================================================================

def load_embedding_model():
    """
    Load the pretrained SentenceTransformer model.

    The first run downloads the model using MODEL_NAME and saves a local copy
    in embedding_model/. Later runs load that local copy. The local copy also
    allows exam.py to work without downloading the model during grading.
    """

    print("\nLOADING EMBEDDING MODEL")
    print("-" * 80)

    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"Loading local model from: {LOCAL_MODEL_DIR}")
        model = SentenceTransformer(LOCAL_MODEL_DIR, device="cpu")
    else:
        print(f"Downloading pretrained model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        model.save(LOCAL_MODEL_DIR)
        print(f"Local model saved to: {LOCAL_MODEL_DIR}")

    return model


def create_and_save_embeddings(documents):
    """
    Embed the document texts and build an exact FAISS search index.

    MiniLM maps each text to a 384-number vector. Normalizing those vectors
    allows FAISS inner-product search to represent cosine similarity.
    """

    print("\nCREATING DOCUMENT EMBEDDINGS")
    print("-" * 80)

    model = load_embedding_model()

    # Only the searchable text is sent to the embedding model.
    document_texts = [document["text"] for document in documents]

    embeddings = model.encode(
        document_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )

    # FAISS requires a contiguous matrix whose numbers use float32.
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    number_of_documents, embedding_dimension = embeddings.shape

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Documents embedded: {number_of_documents}")
    print(f"Numbers per embedding: {embedding_dimension}")

    # Save the embedding matrix explicitly, satisfying the professor's
    # requirement that document embeddings be saved rather than regenerated
    # inside exam.py.
    np.save(EMBEDDINGS_FILE, embeddings)

    # IndexFlatIP performs exact inner-product search. Because the embeddings
    # are normalized, the score is cosine similarity.
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # FAISS stores vectors, but not their restaurant names and source texts.
    # Save the document dictionaries in the same order as the vectors.
    with open(DOCUMENTS_FILE, "wb") as f:
        pickle.dump(documents, f)

    document_type_counts = {
        "restaurant_facts": sum(
            document["document_type"] == "restaurant_facts"
            for document in documents
        ),
        "review": sum(
            document["document_type"] == "review"
            for document in documents
        ),
    }

    metadata = {
        "embedding_model": MODEL_NAME,
        "local_model_directory": "embedding_model",
        "embedding_dimension": embedding_dimension,
        "number_of_documents": number_of_documents,
        "number_of_restaurants": document_type_counts["restaurant_facts"],
        "number_of_clean_reviews": document_type_counts["review"],
        "document_type_counts": document_type_counts,
        "similarity": "cosine similarity using normalized inner product",
        "python_version_required": "3.11",
    }

    config = {
        "model_path": "embedding_model",
        "embeddings_file": "embeddings.npy",
        "faiss_index_file": "faiss.index",
        "documents_file": "documents.pkl",
        "metadata_file": "metadata.json",
        "normalize_embeddings": True,
        "faiss_index_type": "IndexFlatIP",
        "candidate_pool_size": 200,
        "default_result_count": 5,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Every vector must have exactly one matching saved document.
    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index size does not match the number of saved documents."
        )

    print("\nSAVED RETRIEVAL ASSETS")
    print("-" * 80)
    print(f"Embeddings: {EMBEDDINGS_FILE}")
    print(f"FAISS index: {FAISS_INDEX_FILE}")
    print(f"Documents: {DOCUMENTS_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Configuration: {CONFIG_FILE}")
    print(f"Vectors stored in FAISS: {index.ntotal}")

    return model, embeddings, index


# ============================================================================
# 11. Baseline dense retrieval
# ============================================================================

def remove_question_number(question):
    """
    Remove an optional numeric prefix from a question.

    Both of these inputs become the same clean question:
    "0: What are the best restaurants?"
    "What are the best restaurants?"

    This is important because the professor may replace questions.txt with a
    numbered or unnumbered file.
    """

    return re.sub(r"^\s*\d+\s*:\s*", "", question).strip()


def retrieve_candidates(question, model, index, documents, candidate_k=20):
    """
    Retrieve the candidate documents most similar to one question.

    This is baseline dense retrieval. It uses only semantic cosine similarity.
    Later, query-aware logic will rerank these candidates using exact names,
    document type, ratings, and factual metadata.
    """

    clean_question = remove_question_number(question)

    # A question must be embedded with the same model and normalization used
    # for the document embeddings.
    question_embedding = model.encode(
        [clean_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    question_embedding = np.ascontiguousarray(
        question_embedding,
        dtype=np.float32,
    )

    # FAISS returns two matrices:
    # - scores: semantic similarity values
    # - indices: positions of matching documents in documents.pkl
    scores, indices = index.search(question_embedding, candidate_k)

    candidates = []

    for score, document_index in zip(scores[0], indices[0]):
        # FAISS may use -1 when fewer than candidate_k results exist.
        if document_index == -1:
            continue

        # copy() prevents us from modifying the original saved dictionary.
        candidate = documents[int(document_index)].copy()
        candidate["document_index"] = int(document_index)
        candidate["semantic_score"] = float(score)
        candidates.append(candidate)

    return candidates


def test_dense_retrieval(questions, model, index, documents):
    """
    Print baseline results for representative official question categories.

    This function is diagnostic. Its purpose is to reveal where semantic
    retrieval works and where query-aware ranking must be added.
    """

    print("\nBASELINE DENSE RETRIEVAL TEST")
    print("-" * 80)

    # These positions cover recommendation, subjective review, opening-hours,
    # cuisine, and exact-restaurant factual questions.
    test_positions = [0, 1, 2, 15, 20, 22]

    for position in test_positions:
        if position >= len(questions):
            continue

        question = questions[position]
        candidates = retrieve_candidates(
            question=question,
            model=model,
            index=index,
            documents=documents,
            candidate_k=20,
        )

        print(f"\nQuestion: {question}")
        print("Top 5 semantic matches:")

        for rank, candidate in enumerate(candidates[:5], start=1):
            print(
                f"{rank}. {candidate['place_name']} | "
                f"{candidate['document_type']} | "
                f"score={candidate['semantic_score']:.4f}"
            )


# ============================================================================
# 12. Generate and validate grounded answers
# ============================================================================

def generate_and_save_answers(questions, model, index, documents):
    """
    Answer every official question with the query-aware RAG pipeline.

    The output file follows the professor's required format: one question
    line followed immediately by its answer line.
    """

    print("\nQUERY-AWARE QUESTION ANSWERING")
    print("-" * 80)

    results = answer_questions(
        questions=questions,
        model=model,
        index=index,
        documents=documents,
    )

    if len(results) != len(questions):
        raise ValueError("The number of generated answers does not match the questions.")

    for result in results:
        if not result["answer"].strip():
            raise ValueError(
                f"An empty answer was generated for: {result['question']}"
            )

    write_question_answer_file(results, QUESTION_ANSWER_FILE)

    print(f"Answers generated: {len(results)}")
    print(f"Saved to: {QUESTION_ANSWER_FILE}")

    print("\nANSWER PREVIEW")
    print("-" * 80)
    for result in results:
        intent = result["diagnostics"]["analysis"]["intent"]
        print(f"\n{result['question']}")
        print(f"Intent: {intent}")
        print(result["answer"])

    return results


# ============================================================================
# 13. Main execution
# ============================================================================

def main():
    """
    Main execution function.

    Execution order:
    1. Load data.
    2. Check place_id connection.
    3. Merge places and reviews.
    4. Inspect merged data.
    5. Clean merged data.
    6. Build RAG documents.
    7. Create and save document embeddings.
    8. Build and save the FAISS index.
    9. Test baseline semantic retrieval.
    10. Run query-aware retrieval and grounded answer generation.
    11. Save Question-Answer.txt in the required format.
    """

    places_df, reviews_df, questions = load_data()

    print("DATA LOADED")
    print("-" * 80)
    print("places.csv shape:", places_df.shape)
    print("reviews.csv shape:", reviews_df.shape)
    print("number of questions:", len(questions))

    check_place_id_connection(places_df, reviews_df)

    merged_df = merge_places_and_reviews(places_df, reviews_df)

    inspect_merged_data(merged_df)

    cleaned_df = clean_data(merged_df)

    print("\nCLEANING COMPLETE")
    print("-" * 80)
    print("cleaned_df shape:", cleaned_df.shape)

    documents = build_rag_documents(cleaned_df)

    print("\nDOCUMENT BUILDING COMPLETE")
    print("-" * 80)
    print("Number of documents:", len(documents))

    model, embeddings, index = create_and_save_embeddings(documents)

    print("\nEMBEDDING AND INDEX BUILD COMPLETE")
    print("-" * 80)
    print("The saved retrieval system is ready for exam.py.")

    test_dense_retrieval(
        questions=questions,
        model=model,
        index=index,
        documents=documents,
    )

    generate_and_save_answers(
        questions=questions,
        model=model,
        index=index,
        documents=documents,
    )


# ============================================================================
# 14. Run the script
# ============================================================================

# This means:
# if we run this file directly with "python full-code.py",
# then execute main().
if __name__ == "__main__":
    main()
