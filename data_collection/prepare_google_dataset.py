import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_PLACES = BASE_DIR / "google_places.csv"
RAW_REVIEWS = BASE_DIR / "google_reviews.csv"

CLEAN_PLACES = BASE_DIR / "fresh_places.csv"
CLEAN_REVIEWS = BASE_DIR / "fresh_reviews.csv"
REPORT_FILE = BASE_DIR / "google_collection_report.json"

PROFESSOR_PLACES = BASE_DIR / "professor_places_backup.csv"


def main():
    # Load the raw data collected from Google.
    places = pd.read_csv(RAW_PLACES)
    reviews = pd.read_csv(RAW_REVIEWS)

    original_place_count = len(places)
    original_review_count = len(reviews)

    # Remove rows without essential restaurant information.
    places = places.dropna(
        subset=["place_id", "place_name"]
    ).copy()

    places["place_id"] = places["place_id"].astype(str).str.strip()
    places["place_name"] = places["place_name"].astype(str).str.strip()

    places = places[
        (places["place_id"] != "") &
        (places["place_name"] != "")
    ].copy()

    # Keep one metadata row for each restaurant.
    places = places.drop_duplicates(
        subset=["place_id"]
    ).copy()

    # Keep only restaurants located in the municipality of Padova.
    # Google text search may also return restaurants from nearby towns.
    # Padova addresses use the city names Padova/Padua or a 351xx postcode.
    addresses = places["place_address"].fillna("").astype(str)
    padova_mask = addresses.str.contains(
        r"Padova|Padua|\b351\d{2}\b",
        case=False,
        regex=True,
    )
    outside_padova_removed = int((~padova_mask).sum())
    places = places[padova_mask].copy()

    # Remove reviews without an ID or review text.
    reviews = reviews.dropna(
        subset=["place_id", "review"]
    ).copy()

    reviews["place_id"] = reviews["place_id"].astype(str).str.strip()
    reviews["review"] = reviews["review"].astype(str).str.strip()

    reviews = reviews[
        (reviews["place_id"] != "") &
        (reviews["review"] != "")
    ].copy()

    # Keep only reviews connected to a collected restaurant.
    valid_place_ids = set(places["place_id"])

    reviews = reviews[
        reviews["place_id"].isin(valid_place_ids)
    ].copy()

    # Remove repeated review text for the same restaurant.
    before_duplicate_removal = len(reviews)

    reviews = reviews.drop_duplicates(
        subset=["place_id", "review"]
    ).copy()

    duplicates_removed = before_duplicate_removal - len(reviews)

    # Count restaurants represented by at least one review.
    reviewed_place_ids = set(reviews["place_id"])
    restaurants_with_reviews = len(reviewed_place_ids)
    restaurants_without_reviews = len(valid_place_ids - reviewed_place_ids)

    # Every retained restaurant passed the Padova geographic validation.
    padova_address_count = len(places)

    # Compare independently collected IDs with professor data.
    overlap_with_professor = None

    if PROFESSOR_PLACES.exists():
        professor_places = pd.read_csv(PROFESSOR_PLACES)
        professor_ids = set(
            professor_places["place_id"].dropna().astype(str)
        )
        overlap_with_professor = len(valid_place_ids & professor_ids)

    # Save cleaned copies without overwriting the raw Google files.
    places.to_csv(CLEAN_PLACES, index=False)
    reviews.to_csv(CLEAN_REVIEWS, index=False)

    report = {
        "report_created": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source": "Google Places API (New)",
        "city": "Padova, Italy",
        "raw_restaurants": original_place_count,
        "raw_reviews": original_review_count,
        "restaurants_outside_padova_removed": outside_padova_removed,
        "clean_restaurants": len(places),
        "clean_reviews": len(reviews),
        "duplicate_reviews_removed": duplicates_removed,
        "restaurants_with_reviews": restaurants_with_reviews,
        "restaurants_without_reviews": restaurants_without_reviews,
        "padova_address_or_postcode_matches": int(padova_address_count),
        "overlap_with_professor_dataset": overlap_with_professor,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("FRESH DATASET PREPARATION COMPLETE")
    print("-" * 60)
    print("Raw restaurants:", original_place_count)
    print("Clean restaurants:", len(places))
    print("Raw reviews:", original_review_count)
    print("Restaurants outside Padova removed:", outside_padova_removed)
    print("Clean reviews:", len(reviews))
    print("Duplicate reviews removed:", duplicates_removed)
    print("Restaurants with reviews:", restaurants_with_reviews)
    print("Restaurants without reviews:", restaurants_without_reviews)
    print("Padova address/postcode matches:", int(padova_address_count))
    print("Overlap with professor data:", overlap_with_professor)
    print("Saved:", CLEAN_PLACES.name)
    print("Saved:", CLEAN_REVIEWS.name)
    print("Saved:", REPORT_FILE.name)


if __name__ == "__main__":
    main()
