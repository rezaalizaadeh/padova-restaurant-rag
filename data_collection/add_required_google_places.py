import getpass
import urllib.error

from download_google_details import (
    RAW_FILE,
    append_json_line,
    create_csv_files,
    download_place,
    load_downloaded_records,
)


REQUIRED_PLACES = {
    "Bar Fortuna Sas": "ChIJscwimbzbfkcRAcJSXrAFVns",
    "Crazy Pizza": "ChIJ46l-Px7bfkcR4E_2pyKzHvY",
}


def main():
    api_key = getpass.getpass(
        "Paste your Google Places API key: "
    ).strip()

    if not api_key:
        raise ValueError("No API key entered.")

    records = load_downloaded_records()

    for expected_name, place_id in REQUIRED_PLACES.items():
        if place_id in records:
            print(f"Already collected: {expected_name}")
            continue

        print(f"Downloading: {expected_name}")

        try:
            place = download_place(api_key, place_id)

            actual_name = (
                place.get("displayName", {}).get("text", "")
            )

            if not place.get("id") or not actual_name:
                print(f"Invalid response for {expected_name}")
                continue

            append_json_line(RAW_FILE, place)
            records[place["id"]] = place

            print(f"Saved: {actual_name}")

        except urllib.error.HTTPError as error:
            print(
                f"Failed: {expected_name}, HTTP {error.code}"
            )

    review_count = create_csv_files(records)

    print("\nREQUIRED RESTAURANTS COLLECTION COMPLETE")
    print("-" * 60)
    print("Total restaurants:", len(records))
    print("Total reviews:", review_count)


if __name__ == "__main__":
    main()