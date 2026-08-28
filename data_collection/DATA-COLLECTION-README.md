# Fresh Google Places data collection

This directory documents the optional bonus collection performed with Google
Places API (New) on 26 August 2026.

The files in this directory demonstrate independent data collection. Following
the instructor's recommendation, the top-level model, saved embeddings, FAISS
index, and official answers use the instructor-provided dataset instead, so the
results remain directly comparable with the expected course output.

## Results

- 518 unique restaurant identifiers discovered with 83 text-search requests;
- 502 restaurant records collected before geographic validation;
- 2,433 raw review texts returned;
- 51 out-of-city restaurant results removed;
- 451 submitted restaurants explicitly matching Padova, Padua, or a 351xx
  postcode;
- 2,176 submitted clean review texts;
- 445 submitted restaurants represented by review text.

Google Places returned at most five review samples per restaurant. The broader
Google rating count is stored separately in `place_ratings_count`.

## Collection order

Run these commands from this directory:

```bash
python collect_google_places.py
python download_google_details.py
python add_required_google_places.py
python prepare_google_dataset.py
python enrich_fresh_metadata.py
```

The collection programs request the API key with `getpass`; typed characters
are hidden and the key is never written to disk. Search and Place Details
requests have explicit caps and resumable state files.

`prepare_google_dataset.py` creates cleaned files without modifying the raw
Google exports. `enrich_fresh_metadata.py` fills only metadata that is absent
from Google for matching restaurant identifiers.

The top-level `places.csv` and `reviews.csv` are the instructor-provided inputs
used by `full-code.py`. The independently collected strict Padova files are
`fresh_places.csv` and `fresh_reviews.csv` in this directory. API access is not
required by `full-code.py` or `exam.py`.
