import pandas as pd

fresh = pd.read_csv("fresh_places.csv")
professor = pd.read_csv("professor_places_backup.csv")

professor = professor.drop_duplicates("place_id").set_index("place_id")

columns = [
    "has_delivery",
    "has_dine_in",
    "is_reservable",
    "serves_breakfast",
    "serves_lunch",
    "serves_dinner",
    "serves_beer",
    "serves_wine",
    "place_price_level",
]

filled = 0

for column in columns:
    fallback = fresh["place_id"].map(professor[column])

    missing = (
        fresh[column].isna()
        | fresh[column].astype(str).str.strip().eq("")
    )

    usable = missing & fallback.notna()
    filled += int(usable.sum())

    fresh.loc[usable, column] = fallback[usable]

fresh.to_csv("fresh_places.csv", index=False)

print("METADATA ENRICHMENT COMPLETE")
print("Missing metadata values filled:", filled)
print(
    fresh.loc[
        fresh["place_name"] == "Crazy Pizza",
        ["place_name", "has_dine_in", "place_price_level"],
    ]
)