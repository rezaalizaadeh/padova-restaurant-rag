# Data

The system uses restaurant metadata and customer reviews from Padova.

The public repository does not include the complete dataset or generated model artifacts. Local data files should follow these schemas:

## Places

- `place_id`
- `place_name`
- `place_types`
- `place_address`
- `place_average_ratings`
- `place_ratings_count`

## Reviews

- `place_id`
- `review`
- `review_rating`
- `review_publish_time`

The two datasets are joined using `place_id`.
