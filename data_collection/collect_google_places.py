import getpass
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
BASE_DIR = Path(__file__).resolve().parent

STATE_FILE = BASE_DIR / "google_discovery_state.json"
OUTPUT_FILE = BASE_DIR / "google_place_ids.json"

TARGET_PLACES = 520
MAX_SEARCH_REQUESTS = 90

SEARCH_QUERIES = [
    "restaurants in Padova Italy",
    "Italian restaurants in Padova Italy",
    "pizzerias in Padova Italy",
    "trattorias in Padova Italy",
    "osterias in Padova Italy",
    "sushi restaurants in Padova Italy",
    "Japanese restaurants in Padova Italy",
    "Chinese restaurants in Padova Italy",
    "Indian restaurants in Padova Italy",
    "Thai restaurants in Padova Italy",
    "Mexican restaurants in Padova Italy",
    "Mediterranean restaurants in Padova Italy",
    "seafood restaurants in Padova Italy",
    "steakhouses in Padova Italy",
    "burger restaurants in Padova Italy",
    "kebab restaurants in Padova Italy",
    "vegan restaurants in Padova Italy",
    "vegetarian restaurants in Padova Italy",
    "gluten free restaurants in Padova Italy",
    "fine dining restaurants in Padova Italy",
    "family restaurants in Padova Italy",
    "romantic restaurants in Padova Italy",
    "breakfast restaurants in Padova Italy",
    "restaurants in Centro Storico Padova Italy",
    "restaurants in Arcella Padova Italy",
    "restaurants near Prato della Valle Padova Italy",
    "restaurants in Guizza Padova Italy",
    "restaurants in Forcellini Padova Italy",
    "restaurants in Chiesanuova Padova Italy",
    "restaurants in Stanga Padova Italy",
]


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def load_state():
    if not STATE_FILE.exists():
        return {
            "place_ids": [],
            "completed_queries": [],
            "search_requests": 0,
        }

    with open(STATE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def search_places(api_key, query, page_token=None):
    body = {
        "textQuery": query,
        "pageSize": 20,
        "includedType": "restaurant",
        "strictTypeFiltering": True,
        "languageCode": "en",
        "regionCode": "IT",
    }

    if page_token:
        body["pageToken"] = page_token

    request = urllib.request.Request(
        SEARCH_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id,nextPageToken",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    api_key = getpass.getpass(
        "Paste your Google Places API key: "
    ).strip()

    if not api_key:
        raise ValueError("No API key was entered.")

    state = load_state()
    place_ids = set(state["place_ids"])
    completed_queries = set(state["completed_queries"])
    search_requests = int(state["search_requests"])

    print(f"Previously discovered restaurants: {len(place_ids)}")
    print(f"Previous search requests: {search_requests}")

    for query in SEARCH_QUERIES:
        if len(place_ids) >= TARGET_PLACES:
            break

        if search_requests >= MAX_SEARCH_REQUESTS:
            break

        if query in completed_queries:
            continue

        print(f"\nSearching: {query}")
        page_token = None

        for page_number in range(1, 4):
            if search_requests >= MAX_SEARCH_REQUESTS:
                break

            result = search_places(api_key, query, page_token)
            search_requests += 1

            for place in result.get("places", []):
                place_id = place.get("id")

                if place_id:
                    place_ids.add(place_id)

            state["place_ids"] = sorted(place_ids)
            state["completed_queries"] = sorted(completed_queries)
            state["search_requests"] = search_requests
            save_json(STATE_FILE, state)

            print(
                f"Page {page_number}: "
                f"{len(place_ids)} unique restaurants found"
            )

            if len(place_ids) >= TARGET_PLACES:
                break

            page_token = result.get("nextPageToken")

            if not page_token:
                break

            time.sleep(2)

        completed_queries.add(query)

        state["place_ids"] = sorted(place_ids)
        state["completed_queries"] = sorted(completed_queries)
        state["search_requests"] = search_requests
        save_json(STATE_FILE, state)

    final_ids = sorted(place_ids)

    save_json(OUTPUT_FILE, final_ids)

    print("\nDISCOVERY COMPLETE")
    print("-" * 60)
    print("Unique restaurant IDs:", len(final_ids))
    print("Search requests used:", search_requests)
    print("Saved to:", OUTPUT_FILE.name)
    print("No restaurant-detail or review requests were made.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        print("API ERROR:", error.code)
        print(error.read().decode("utf-8"))
    except Exception as error:
        print("ERROR:", error)