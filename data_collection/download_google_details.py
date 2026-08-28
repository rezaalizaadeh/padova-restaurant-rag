import csv
import getpass
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

IDS_FILE = BASE_DIR / "google_place_ids.json"
RAW_FILE = BASE_DIR / "google_place_details.jsonl"
STATE_FILE = BASE_DIR / "google_detail_state.json"
ERROR_FILE = BASE_DIR / "google_detail_errors.jsonl"

PLACES_OUTPUT = BASE_DIR / "google_places.csv"
REVIEWS_OUTPUT = BASE_DIR / "google_reviews.csv"

TARGET_RESTAURANTS = 500
MAX_DETAIL_REQUESTS = 500

FIELD_MASK = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "types",
    "primaryType",
    "location",
    "rating",
    "userRatingCount",
    "reviews",
    "websiteUri",
    "nationalPhoneNumber",
    "priceLevel",
    "regularOpeningHours",
    "delivery",
    "dineIn",
    "reservable",
    "servesBreakfast",
    "servesLunch",
    "servesDinner",
    "servesBeer",
    "servesWine",
])


def load_json(path, default):
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def append_json_line(path, value):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")
        file.flush()


def load_downloaded_records():
    records = {}

    if not RAW_FILE.exists():
        return records

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                place_id = record.get("id")

                if place_id:
                    records[place_id] = record
            except json.JSONDecodeError:
                continue

    return records


def download_place(api_key, place_id):
    encoded_id = urllib.parse.quote(place_id, safe="")

    url = (
        f"https://places.googleapis.com/v1/places/{encoded_id}"
        "?languageCode=en&regionCode=IT"
    )

    request = urllib.request.Request(
        url,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": FIELD_MASK,
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def review_text(review):
    text_data = review.get("text") or review.get("originalText") or {}
    return str(text_data.get("text", "")).strip()


def create_csv_files(records):
    place_columns = [
        "place_id",
        "place_name",
        "place_types",
        "place_address",
        "place_average_ratings",
        "place_ratings_count",
        "place_reviews_count",
        "place_location",
        "place_website",
        "place_phone_number",
        "place_price_level",
        "place_primary_type",
        "has_delivery",
        "has_dine_in",
        "is_reservable",
        "serves_breakfast",
        "serves_lunch",
        "serves_dinner",
        "serves_beer",
        "serves_wine",
        "places_opening_hours",
    ]

    review_columns = [
        "place_id",
        "review",
        "review_rating",
        "review_publish_time",
        "review_publish_time_str",
    ]

    review_rows = []

    with open(
        PLACES_OUTPUT, "w", newline="", encoding="utf-8"
    ) as places_file:
        writer = csv.DictWriter(places_file, fieldnames=place_columns)
        writer.writeheader()

        for place in records.values():
            place_id = place.get("id", "")
            reviews = place.get("reviews", [])

            writer.writerow({
                "place_id": place_id,
                "place_name": place.get("displayName", {}).get("text", ""),
                "place_types": json.dumps(
                    place.get("types", []), ensure_ascii=False
                ),
                "place_address": place.get("formattedAddress", ""),
                "place_average_ratings": place.get("rating", ""),
                "place_ratings_count": place.get("userRatingCount", ""),
                "place_reviews_count": len(reviews),
                "place_location": json.dumps(
                    place.get("location", {}), ensure_ascii=False
                ),
                "place_website": place.get("websiteUri", ""),
                "place_phone_number": place.get(
                    "nationalPhoneNumber", ""
                ),
                "place_price_level": place.get("priceLevel", ""),
                "place_primary_type": place.get("primaryType", ""),
                "has_delivery": place.get("delivery", ""),
                "has_dine_in": place.get("dineIn", ""),
                "is_reservable": place.get("reservable", ""),
                "serves_breakfast": place.get("servesBreakfast", ""),
                "serves_lunch": place.get("servesLunch", ""),
                "serves_dinner": place.get("servesDinner", ""),
                "serves_beer": place.get("servesBeer", ""),
                "serves_wine": place.get("servesWine", ""),
                "places_opening_hours": json.dumps(
                    place.get(
                        "regularOpeningHours", {}
                    ).get("weekdayDescriptions", []),
                    ensure_ascii=False,
                ),
            })

            for review in reviews:
                text = review_text(review)

                if not text:
                    continue

                review_rows.append({
                    "place_id": place_id,
                    "review": text,
                    "review_rating": review.get("rating", ""),
                    "review_publish_time": review.get(
                        "publishTime", ""
                    ),
                    "review_publish_time_str": review.get(
                        "relativePublishTimeDescription", ""
                    ),
                })

    with open(
        REVIEWS_OUTPUT, "w", newline="", encoding="utf-8"
    ) as reviews_file:
        writer = csv.DictWriter(reviews_file, fieldnames=review_columns)
        writer.writeheader()
        writer.writerows(review_rows)

    return len(review_rows)


def main():
    place_ids = load_json(IDS_FILE, [])

    if len(place_ids) < TARGET_RESTAURANTS:
        raise RuntimeError(
            f"Only {len(place_ids)} restaurant IDs are available."
        )

    api_key = getpass.getpass(
        "Paste your Google Places API key: "
    ).strip()

    if not api_key:
        raise ValueError("No API key was entered.")

    records = load_downloaded_records()

    state = load_json(
        STATE_FILE,
        {
            "attempted_ids": [],
            "detail_requests": 0,
        },
    )

    attempted_ids = set(state["attempted_ids"])
    detail_requests = int(state["detail_requests"])

    print("Already downloaded:", len(records))
    print("Previous detail requests:", detail_requests)
    print("Maximum new dataset size:", TARGET_RESTAURANTS)

    confirmation = input(
        '\nType "DOWNLOAD 500" to begin: '
    ).strip()

    if confirmation != "DOWNLOAD 500":
        print("Collection cancelled.")
        return

    for place_id in place_ids:
        if len(records) >= TARGET_RESTAURANTS:
            break

        if detail_requests >= MAX_DETAIL_REQUESTS:
            break

        if place_id in attempted_ids:
            continue

        # Record the attempt before calling Google.
        attempted_ids.add(place_id)
        detail_requests += 1

        state["attempted_ids"] = sorted(attempted_ids)
        state["detail_requests"] = detail_requests
        save_json(STATE_FILE, state)

        try:
            place = download_place(api_key, place_id)
            append_json_line(RAW_FILE, place)

            downloaded_id = place.get("id", place_id)
            records[downloaded_id] = place

            if len(records) % 10 == 0:
                print(
                    f"Downloaded {len(records)}/"
                    f"{TARGET_RESTAURANTS} restaurants"
                )

        except urllib.error.HTTPError as error:
            error_text = error.read().decode("utf-8")

            append_json_line(
                ERROR_FILE,
                {
                    "place_id": place_id,
                    "status": error.code,
                    "error": error_text,
                },
            )

            print(
                f"Request {detail_requests} failed "
                f"with HTTP {error.code}"
            )

            if error.code in {403, 429}:
                print("Stopping for API or quota protection.")
                break

        except Exception as error:
            append_json_line(
                ERROR_FILE,
                {
                    "place_id": place_id,
                    "error": str(error),
                },
            )

            print(
                f"Request {detail_requests} failed: {error}"
            )

        time.sleep(0.15)

    review_count = create_csv_files(records)

    print("\nDOWNLOAD COMPLETE")
    print("-" * 60)
    print("Successful restaurants:", len(records))
    print("Detail requests used:", detail_requests)
    print("Reviews with text:", review_count)
    print("Saved:", PLACES_OUTPUT.name)
    print("Saved:", REVIEWS_OUTPUT.name)


if __name__ == "__main__":
    main()