"""
Query-aware retrieval and grounded answer generation for the Padova RAG system.

This module is shared by full-code.py and exam.py.  It never creates document
embeddings.  It only searches an already-built FAISS index and uses the saved
document metadata to construct evidence-based answers.
"""

import math
import re
from collections import defaultdict

import numpy as np


ASPECT_KEYWORDS = {
    "ambiance": (
        "ambiance", "ambience", "atmosphere", "cozy", "cosy", "decor",
        "interior", "vibe", "elegant", "quiet",
    ),
    "service": (
        "service", "staff", "waiter", "waitress", "friendly", "kind",
        "welcoming", "attentive", "professional",
    ),
    "vegan": ("vegan", "plant based", "plant-based"),
    "gluten_free": ("gluten free", "gluten-free", "celiac", "coeliac"),
    "keto": ("keto", "ketogenic", "low carb", "low-carb"),
    "halal": ("halal",),
    "hygiene": (
        "hygiene", "hygienic", "clean", "cleanliness", "spotless",
    ),
    "affordable": (
        "affordable", "cheap", "inexpensive", "good value", "value for money",
        "reasonable price", "fair price", "budget",
    ),
    "outdoor": (
        "outdoor", "outdoors", "outdoor seating", "sat outside",
        "sitting outside", "tables outside", "terrace", "patio", "garden",
        "open air", "al fresco",
    ),
    "live_music": (
        "live music", "live band", "concert", "musician", "music night",
    ),
    "romantic": (
        "romantic", "date night", "intimate", "cozy", "cosy", "elegant",
        "quiet atmosphere",
    ),
    "family": (
        "family-friendly", "family friendly", "family gathering",
        "with my family", "families with children", "children", "kids",
        "kid-friendly", "kids area",
    ),
    "celebration": (
        "birthday", "celebration", "party", "special occasion",
    ),
    "central": (
        "city center", "city centre", "downtown", "historic center",
        "historical center", "old town", "center of padova",
        "centre of padova", "central padova",
    ),
    "italian": (
        "italian", "pasta", "risotto", "traditional cuisine",
    ),
    "sushi": ("sushi", "japanese", "sashimi", "nigiri", "maki"),
}

POSITIVE_EVIDENCE_KEYWORDS = (
    "excellent", "amazing", "delicious", "great", "outstanding",
    "highly recommended", "highly recommend", "friendly", "wonderful",
    "fantastic", "tasty",
)


CUISINE_TYPES = {
    "italian": ("italian restaurant", "pizza restaurant"),
    "sushi": ("sushi restaurant", "japanese restaurant"),
}


PROPERTY_LABELS = {
    "has_delivery": "delivery services",
    "has_dine_in": "dine-in",
    "is_reservable": "reservations",
    "serves_breakfast": "breakfast",
    "serves_lunch": "lunch",
    "serves_dinner": "dinner",
    "serves_beer": "beer",
    "serves_wine": "wine",
}


def remove_question_number(question):
    """Remove an optional prefix such as ``12:`` from a question."""

    return re.sub(r"^\s*\d+\s*:\s*", "", str(question)).strip()


def normalize_text(value):
    """Create lowercase alphanumeric text for robust comparisons."""

    value = str(value).casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def keyword_hits(text, keywords):
    """Count how many distinct keywords occur in normalized text."""

    normalized = normalize_text(text)
    return sum(normalize_text(keyword) in normalized for keyword in keywords)


def truncate_text(text, limit=180):
    """Return a readable evidence excerpt without cutting the last word."""

    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text

    shortened = text[:limit].rsplit(" ", 1)[0]
    return shortened + "…"


def build_restaurant_profiles(documents):
    """
    Group the saved fact/review documents by restaurant.

    A profile contains one factual document and all available review
    documents.  Grouping allows the ranker to return unique restaurants rather
    than repeating several reviews from the same place.
    """

    profiles = {}

    for document in documents:
        place_id = str(document["place_id"])

        if place_id not in profiles:
            profiles[place_id] = {
                "place_id": place_id,
                "place_name": document["place_name"],
                "facts": None,
                "reviews": [],
            }

        if document["document_type"] == "restaurant_facts":
            profiles[place_id]["facts"] = document
        else:
            profiles[place_id]["reviews"].append(document)

    # Every restaurant should have a factual document.  Falling back to a
    # review keeps the program robust if a future dataset is incomplete.
    for profile in profiles.values():
        if profile["facts"] is None and profile["reviews"]:
            profile["facts"] = profile["reviews"][0]

    return profiles


def find_exact_restaurant(question, profiles):
    """
    Find the longest restaurant name explicitly present in the question.

    Longest-first matching avoids selecting a shorter name such as
    ``Bar Fortuna`` when the question contains ``Bar Fortuna Sas``.
    """

    normalized_question = f" {normalize_text(question)} "
    names = sorted(
        (
            (normalize_text(profile["place_name"]), profile["place_id"])
            for profile in profiles.values()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for normalized_name, place_id in names:
        if len(normalized_name) < 4:
            continue

        if f" {normalized_name} " in normalized_question:
            return place_id

    return None


def analyze_question(question, profiles):
    """Classify the evidence and ranking strategy needed for one question."""

    clean_question = remove_question_number(question)
    normalized = normalize_text(clean_question)
    exact_place_id = find_exact_restaurant(clean_question, profiles)

    property_name = None
    if "deliver" in normalized:
        property_name = "has_delivery"
    elif "dine in" in normalized or "dinein" in normalized:
        property_name = "has_dine_in"
    elif "reserv" in normalized:
        property_name = "is_reservable"
    elif "breakfast" in normalized:
        property_name = "serves_breakfast"
    elif "lunch" in normalized:
        property_name = "serves_lunch"
    elif "dinner" in normalized:
        property_name = "serves_dinner"
    elif re.search(r"\bbeer\b", normalized):
        property_name = "serves_beer"
    elif re.search(r"\bwine\b", normalized):
        property_name = "serves_wine"

    aspect = None
    aspect_patterns = [
        ("live_music", ("live music", "live band", "concert")),
        ("gluten_free", ("gluten free", "gluten-free", "celiac")),
        ("affordable", ("affordable", "budget", "cheap", "value for money")),
        ("ambiance", ("ambiance", "ambience", "atmosphere")),
        ("service", ("customer service", "best service", "good service")),
        ("hygiene", ("hygiene", "cleanliness", "hygienic")),
        ("outdoor", ("outdoor", "outside seating", "terrace", "patio")),
        ("romantic", ("romantic", "date night")),
        ("family", ("family", "children", "kids")),
        ("celebration", ("birthday", "celebration", "special occasion")),
        ("vegan", ("vegan", "plant based", "plant-based")),
        ("keto", ("keto", "ketogenic", "low carb")),
        ("halal", ("halal",)),
        ("sushi", ("sushi", "japanese")),
        ("italian", ("italian",)),
        ("central", ("downtown", "city center", "city centre", "old town")),
    ]

    for aspect_name, patterns in aspect_patterns:
        if any(normalize_text(pattern) in normalized for pattern in patterns):
            aspect = aspect_name
            break

    if "24 hour" in normalized or "24hour" in normalized:
        intent = "twenty_four_hour"
    elif property_name and exact_place_id:
        intent = "exact_fact"
    elif exact_place_id:
        intent = "exact_recommendation"
    elif "near landmark" in normalized or re.search(r"\blandmark\b", normalized):
        intent = "location_placeholder"
    elif re.search(r"\bneighborhood\b", normalized):
        intent = "neighborhood_placeholder"
    elif aspect == "central":
        intent = "location_evidence"
    elif aspect == "hygiene":
        intent = "unsupported_hygiene"
    elif aspect in {"outdoor", "live_music"}:
        intent = "review_evidence"
    elif aspect == "affordable":
        intent = "affordable"
    elif aspect:
        intent = "aspect"
    else:
        intent = "overall"

    return {
        "question": clean_question,
        "normalized_question": normalized,
        "exact_place_id": exact_place_id,
        "property_name": property_name,
        "aspect": aspect,
        "intent": intent,
    }


def is_negative_evidence(text, aspect, review_rating):
    """Reject obvious counter-evidence from recommendation answers."""

    normalized = normalize_text(text)

    # Very low-rated reviews should not support a positive recommendation.
    if float(review_rating or 0.0) <= 2.0:
        return True

    negative_patterns = {
        "gluten_free": (
            "not a great gluten free",
            "not gluten free",
            "not gluten-free",
            "gluten free advertisement is a joke",
            "gluten-free advertisement is a joke",
        ),
        "vegan": ("no vegan", "not vegan", "without vegan"),
        "halal": ("not halal", "no halal"),
        "family": ("treated decently", "family with kids for minutes"),
        "celebration": (
            "staff refused",
            "service haphazard",
            "rude to customers",
            "denied us",
        ),
        "affordable": ("overpriced", "very expensive", "not worth the price"),
        "central": (
            "outside downtown",
            "away from downtown",
            "far from the city center",
        ),
    }

    return any(
        normalize_text(pattern) in normalized
        for pattern in negative_patterns.get(aspect, ())
    )


def is_positive_evidence(text, aspect):
    """Require stronger wording for aspects prone to accidental matches."""

    if aspect != "celebration":
        return True

    normalized = normalize_text(text)
    celebration_phrases = (
        "celebrated",
        "celebrate",
        "having my birthday",
        "birthday celebration",
        "birthday lunch",
        "party there",
        "candle",
        "great place to have parties",
    )
    return any(
        normalize_text(phrase) in normalized
        for phrase in celebration_phrases
    )


def dense_retrieve(question, model, index, documents, candidate_k=200):
    """Retrieve a broad semantic candidate pool from the saved FAISS index."""

    clean_question = remove_question_number(question)
    question_embedding = model.encode(
        [clean_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
        show_progress_bar=False,
    )
    question_embedding = np.ascontiguousarray(
        question_embedding,
        dtype=np.float32,
    )

    candidate_k = min(int(candidate_k), int(index.ntotal))
    scores, indices = index.search(question_embedding, candidate_k)
    candidates = []

    for score, document_index in zip(scores[0], indices[0]):
        if int(document_index) < 0:
            continue

        candidate = documents[int(document_index)].copy()
        candidate["document_index"] = int(document_index)
        candidate["semantic_score"] = float(score)
        candidates.append(candidate)

    return candidates


def normalized_rating_score(rating):
    """Map the observed 3–5 star range approximately onto 0–1."""

    return max(0.0, min(1.0, (float(rating) - 3.0) / 2.0))


def bayesian_quality_score(rating, rating_count, global_mean=4.28, prior=100):
    """
    Estimate rating quality while reducing the advantage of tiny samples.

    A 5.0 rating based on 50 ratings should not automatically outrank a 4.8
    rating based on thousands.  The prior pulls small samples toward the
    dataset mean.
    """

    rating = float(rating)
    rating_count = max(0, int(rating_count))
    adjusted = (
        rating_count / (rating_count + prior) * rating
        + prior / (rating_count + prior) * global_mean
    )
    return normalized_rating_score(adjusted)


def rank_restaurants(analysis, dense_candidates, profiles, top_k=5):
    """
    Aggregate document candidates, apply query-aware evidence, and rank unique
    restaurants.
    """

    semantic_by_place = defaultdict(float)
    semantic_by_index = {}

    for candidate in dense_candidates:
        place_id = str(candidate["place_id"])
        semantic_by_place[place_id] = max(
            semantic_by_place[place_id],
            float(candidate["semantic_score"]),
        )
        semantic_by_index[int(candidate["document_index"])] = float(
            candidate["semantic_score"]
        )

    max_rating_count = max(
        int(profile["facts"].get("place_ratings_count", 0))
        for profile in profiles.values()
    )

    if analysis["exact_place_id"]:
        candidate_place_ids = [analysis["exact_place_id"]]
    elif analysis["aspect"]:
        # Explicit structured/lexical evidence may occur outside FAISS's
        # semantic candidate pool, so aspect queries inspect all 513 compact
        # restaurant profiles and still use dense similarity when available.
        candidate_place_ids = list(profiles)
    else:
        candidate_place_ids = list(semantic_by_place)

    aspect = analysis["aspect"]
    keywords = ASPECT_KEYWORDS.get(aspect, ())
    ranked = []

    for place_id in candidate_place_ids:
        profile = profiles[place_id]
        facts = profile["facts"]
        rating = float(facts.get("place_average_ratings", 0.0))
        rating_count = int(facts.get("place_ratings_count", 0))
        semantic = semantic_by_place.get(place_id, 0.0)
        quality = bayesian_quality_score(rating, rating_count)
        popularity = math.log1p(rating_count) / math.log1p(max_rating_count)

        supporting_reviews = []
        for review in profile["reviews"]:
            review_text = review.get("review_text") or review.get("text", "")
            hits = keyword_hits(review_text, keywords) if keywords else 0
            if hits and not is_negative_evidence(
                review_text,
                aspect,
                review.get("review_rating"),
            ) and is_positive_evidence(review_text, aspect):
                supporting_reviews.append((review, hits))

        # Count distinct supporting reviews, not only distinct words.  Several
        # independent reviews are stronger evidence than one repeated phrase.
        aspect_hits = sum(min(2, hits) for _, hits in supporting_reviews)
        aspect_score = min(1.0, len(supporting_reviews) / 3.0)

        type_text = (
            f"{facts.get('place_primary_type', '')} "
            f"{facts.get('place_types', '')}"
        )
        type_matches = CUISINE_TYPES.get(aspect, ())
        type_score = (
            1.0
            if type_matches
            and any(
                normalize_text(type_name) in normalize_text(type_text)
                for type_name in type_matches
            )
            else 0.0
        )

        price_text = normalize_text(facts.get("place_price_level", ""))
        inexpensive = 1.0 if "inexpensive" in price_text else 0.0

        # Use aspect-specific review evidence when possible.  The fallback is
        # the highest-rated review for that restaurant.
        evidence_review = None
        if keywords and supporting_reviews:
            best_tuple = (-1.0, None)
            for review, hits in supporting_reviews:
                review_text = review.get("review_text") or review.get("text", "")
                lexical = min(1.0, hits / 2.0)
                review_score = float(review.get("review_rating") or 0.0) / 5.0
                value = 0.65 * lexical + 0.35 * review_score
                if value > best_tuple[0]:
                    best_tuple = (value, review)
            _, evidence_review = best_tuple
        elif profile["reviews"]:
            evidence_review = max(
                profile["reviews"],
                key=lambda review: (
                    float(review.get("review_rating") or 0.0),
                    keyword_hits(
                        review.get("review_text") or review.get("text", ""),
                        POSITIVE_EVIDENCE_KEYWORDS,
                    ),
                ),
            )

        intent = analysis["intent"]
        if intent == "overall":
            final_score = 0.25 * semantic + 0.55 * quality + 0.20 * popularity
        elif intent == "affordable":
            final_score = (
                0.25 * semantic
                + 0.25 * quality
                + 0.30 * inexpensive
                + 0.20 * aspect_score
            )
        elif aspect in {"italian", "sushi"}:
            final_score = (
                0.30 * semantic
                + 0.30 * quality
                + 0.30 * max(type_score, aspect_score)
                + 0.10 * popularity
            )
        else:
            effective_semantic = max(
                semantic,
                0.40 if aspect_hits > 0 else semantic,
            )
            final_score = (
                0.30 * effective_semantic
                + 0.30 * quality
                + 0.30 * aspect_score
                + 0.10 * popularity
            )

        if analysis["exact_place_id"] == place_id:
            final_score += 1.0

        ranked.append(
            {
                "place_id": place_id,
                "place_name": profile["place_name"],
                "facts": facts,
                "evidence_review": evidence_review,
                "semantic_score": semantic,
                "quality_score": quality,
                "aspect_hits": aspect_hits,
                "type_score": type_score,
                "final_score": final_score,
            }
        )

    # Queries relying on review-only evidence should not present restaurants
    # with no matching evidence as confirmed recommendations.
    if analysis["intent"] in {
        "review_evidence",
        "unsupported_hygiene",
        "location_evidence",
    }:
        ranked = [item for item in ranked if item["aspect_hits"] > 0]

    if analysis["intent"] == "aspect" and aspect in {
        "vegan",
        "gluten_free",
        "keto",
        "halal",
    }:
        ranked = [
            item
            for item in ranked
            if item["aspect_hits"] > 0 or item["type_score"] > 0
        ]

    if analysis["intent"] == "aspect" and aspect not in {
        "italian",
        "sushi",
        "vegan",
        "gluten_free",
        "keto",
        "halal",
    }:
        ranked = [item for item in ranked if item["aspect_hits"] > 0]

    if analysis["intent"] == "affordable":
        # "High-quality" is part of the question, so exclude restaurants whose
        # average rating is below the dataset's typical quality range.
        ranked = [
            item
            for item in ranked
            if float(item["facts"].get("place_average_ratings", 0.0)) >= 4.3
        ]

    if (
        analysis["intent"] in {"overall", "affordable"}
        and "restaurant" in analysis["normalized_question"]
    ):
        ranked = [
            item
            for item in ranked
            if "restaurant" in normalize_text(
                f"{item['facts'].get('place_primary_type', '')} "
                f"{item['facts'].get('place_types', '')}"
            )
        ]

    if analysis["intent"] == "unsupported_hygiene":
        ranked = [
            item
            for item in ranked
            if float(item["facts"].get("place_average_ratings", 0.0)) >= 4.3
        ]

    if aspect in {"italian", "sushi"}:
        ranked = [
            item
            for item in ranked
            if item["type_score"] > 0 or item["aspect_hits"] > 0
        ]

    ranked.sort(
        key=lambda item: (
            item["final_score"],
            item["facts"].get("place_average_ratings", 0.0),
            item["facts"].get("place_ratings_count", 0),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def format_rating(item):
    """Format a restaurant's rating and rating-count evidence."""

    facts = item["facts"]
    return (
        f"{float(facts.get('place_average_ratings', 0.0)):.1f}/5 "
        f"from {int(facts.get('place_ratings_count', 0))} ratings"
    )


def evidence_excerpt(item, aspect=None):
    """Return a compact customer-review excerpt for a ranked restaurant."""

    review = item.get("evidence_review")
    if not review:
        return ""

    text = " ".join(
        str(review.get("review_text") or review.get("text", "")).split()
    )
    keywords = ASPECT_KEYWORDS.get(aspect, ())
    # Start near the relevant phrase so the displayed evidence actually shows
    # why the restaurant was selected.
    for keyword in keywords:
        position = text.casefold().find(str(keyword).casefold())
        if position >= 0:
            start = max(0, position - 65)
            end = min(len(text), position + len(keyword) + 115)
            excerpt = text[start:end]
            if start > 0:
                excerpt = "…" + excerpt.lstrip()
            if end < len(text):
                excerpt = excerpt.rstrip() + "…"
            return excerpt

    return truncate_text(text)


def answer_exact_fact(analysis, profiles):
    """Answer a factual property question using the exact restaurant profile."""

    profile = profiles[analysis["exact_place_id"]]
    facts = profile["facts"]
    property_name = analysis["property_name"]
    value = facts.get(property_name, "Unknown")
    label = PROPERTY_LABELS[property_name]
    name = profile["place_name"]

    if value == "Yes":
        return f"Yes. According to the dataset metadata, {name} offers {label}."
    if value == "No":
        return f"No. According to the dataset metadata, {name} does not offer {label}."

    return (
        f"The dataset does not provide a confirmed value for whether "
        f"{name} offers {label}."
    )


def answer_twenty_four_hour(profiles):
    """Answer from explicit opening-hours metadata rather than similarity."""

    matches = []

    for profile in profiles.values():
        facts = profile["facts"]
        hours = str(facts.get("places_opening_hours", ""))
        if "open 24 hours" in hours.casefold():
            matches.append((profile["place_name"], hours))

    if not matches:
        return (
            "No restaurant in the dataset is explicitly listed as open for "
            "24 hours. This conclusion is based on the stored opening-hours "
            "metadata."
        )

    descriptions = [
        f"{name} ({hours})"
        for name, hours in sorted(matches, key=lambda item: item[0])
    ]
    return (
        "The opening-hours metadata identifies: "
        + "; ".join(descriptions)
        + ". An 'Open 24 hours' entry may apply only on the listed days, so "
        "this should not be interpreted automatically as open 24/7."
    )


def answer_exact_recommendation(analysis, ranked, profiles):
    """Answer an opinion question about one explicitly named restaurant."""

    profile = profiles[analysis["exact_place_id"]]
    facts = profile["facts"]
    name = profile["place_name"]
    rating = float(facts.get("place_average_ratings", 0.0))
    count = int(facts.get("place_ratings_count", 0))
    aspect = analysis["aspect"]

    if aspect == "affordable":
        price = normalize_text(facts.get("place_price_level", ""))
        price_statement = (
            "is marked as inexpensive"
            if "inexpensive" in price
            else "has no confirmed inexpensive price classification"
        )
        evidence = evidence_excerpt(ranked[0], aspect) if ranked else ""
        answer = (
            f"{name} {price_statement} and has an average rating of "
            f"{rating:.1f}/5 from {count} ratings."
        )
        if evidence:
            answer += f' A relevant review says: "{evidence}"'
        return answer

    return (
        f"{name} has an average rating of {rating:.1f}/5 from {count} ratings. "
        "The available restaurant metadata and reviews should be considered "
        "together when deciding whether it fits the requested preference."
    )


def answer_ranked_list(analysis, ranked):
    """Compose a concise source-grounded recommendation answer."""

    if not ranked:
        aspect_label = (analysis["aspect"] or "requested feature").replace("_", " ")
        return (
            f"The dataset does not contain enough explicit evidence to confirm "
            f"restaurants for {aspect_label}. I would not infer this feature "
            "without supporting metadata or review text."
        )

    entries = []
    for item in ranked[:3]:
        entry = f"{item['place_name']} ({format_rating(item)})"
        excerpt = evidence_excerpt(item, analysis["aspect"])
        if excerpt:
            entry += f', supported by the review excerpt: "{excerpt}"'
        entries.append(entry)

    prefix = "Based on the retrieved metadata and customer-review evidence, "

    if analysis["intent"] == "unsupported_hygiene":
        prefix = (
            "The dataset has no official hygiene-rating field. Based only on "
            "reviews that mention cleanliness or hygiene, possible matches are "
        )
    elif analysis["intent"] == "review_evidence":
        feature = analysis["aspect"].replace("_", " ")
        prefix = (
            f"The dataset has no structured {feature} field. Based on explicit "
            "customer-review mentions, the strongest matches are "
        )
    elif analysis["intent"] == "location_placeholder":
        prefix = (
            "The question contains the placeholder 'Landmark', so proximity "
            "cannot be verified without a real location. As citywide "
            "high-quality alternatives, the ranking returns "
        )
    elif analysis["intent"] == "neighborhood_placeholder":
        prefix = (
            "The question contains the placeholder 'Neighborhood', so a "
            "neighborhood-specific answer cannot be verified. As citywide "
            "alternatives, the ranking returns "
        )
    elif analysis["intent"] == "location_evidence":
        prefix = (
            "The dataset has no formal downtown boundary. Based on customer "
            "reviews that explicitly mention the city center or downtown, "
            "strong matches are "
        )

    return prefix + "; ".join(entries) + "."


def answer_question(question, model, index, documents, profiles=None, top_k=5):
    """
    Run query analysis, dense retrieval, query-aware ranking, and generation.

    Returns both the final answer and diagnostic information used for testing.
    """

    if profiles is None:
        profiles = build_restaurant_profiles(documents)

    analysis = analyze_question(question, profiles)

    if analysis["intent"] == "exact_fact":
        answer = answer_exact_fact(analysis, profiles)
        return answer, {"analysis": analysis, "ranked": []}

    if analysis["intent"] == "twenty_four_hour":
        answer = answer_twenty_four_hour(profiles)
        return answer, {"analysis": analysis, "ranked": []}

    candidates = dense_retrieve(
        question=question,
        model=model,
        index=index,
        documents=documents,
        candidate_k=200,
    )
    ranked = rank_restaurants(
        analysis=analysis,
        dense_candidates=candidates,
        profiles=profiles,
        top_k=top_k,
    )

    if analysis["intent"] == "exact_recommendation":
        answer = answer_exact_recommendation(analysis, ranked, profiles)
    else:
        answer = answer_ranked_list(analysis, ranked)

    return answer, {"analysis": analysis, "ranked": ranked}


def answer_questions(questions, model, index, documents):
    """Answer a list of numbered or unnumbered questions."""

    profiles = build_restaurant_profiles(documents)
    results = []

    for question in questions:
        answer, diagnostics = answer_question(
            question=question,
            model=model,
            index=index,
            documents=documents,
            profiles=profiles,
        )
        results.append(
            {
                "question": str(question).strip(),
                "answer": answer,
                "diagnostics": diagnostics,
            }
        )

    return results


def write_question_answer_file(results, output_path):
    """Write the professor's required question-then-answer text format."""

    with open(output_path, "w", encoding="utf-8") as file:
        for position, result in enumerate(results):
            raw_question = result["question"]
            if re.match(r"^\s*\d+\s*:", raw_question):
                question_line = raw_question
            else:
                question_line = f"{position}: {raw_question}"

            file.write(question_line.strip() + "\n")
            file.write(result["answer"].strip() + "\n")
