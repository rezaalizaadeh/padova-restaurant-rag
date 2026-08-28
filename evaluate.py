"""Small reproducible validation suite for the saved Padova RAG system."""

from exam import load_saved_system
from qa_engine import answer_question, build_restaurant_profiles


def main():
    model, index, documents = load_saved_system()
    profiles = build_restaurant_profiles(documents)

    checks = []

    def run(question):
        return answer_question(
            question,
            model,
            index,
            documents,
            profiles=profiles,
        )

    answer, diagnostics = run(
        "Does Bar Fortuna Sas offer delivery services?"
    )
    checks.append(
        (
            "exact-name delivery fact",
            diagnostics["analysis"]["intent"] == "exact_fact"
            and answer.startswith("No.")
            and "Bar Fortuna Sas" in answer,
        )
    )

    answer, diagnostics = run("Does Crazy Pizza offer dine-in options?")
    checks.append(
        (
            "exact-name dine-in fact",
            diagnostics["analysis"]["intent"] == "exact_fact"
            and answer.startswith("No.")
            and "Crazy Pizza" in answer,
        )
    )

    answer, _ = run("Does Rasa multi cuisine restaurant offer delivery?")
    checks.append(("positive delivery fact", answer.startswith("Yes.")))

    answer, _ = run("Are there any 24-hour restaurants in the city?")
    checks.append(
        (
            "opening-hours grounding",
            "opening-hours metadata identifies" in answer
            and "Open 24 hours" in answer,
        )
    )

    answer, diagnostics = run(
        "Which restaurant has the best ambiance in the city?"
    )
    ambiance_results = diagnostics["ranked"]
    checks.append(
        (
            "ambiance evidence coverage",
            bool(ambiance_results)
            and all(item["aspect_hits"] > 0 for item in ambiance_results),
        )
    )

    answer, diagnostics = run("What are the best restaurants in the city?")
    place_ids = [item["place_id"] for item in diagnostics["ranked"]]
    checks.append(
        (
            "unique restaurant aggregation",
            len(place_ids) == len(set(place_ids)),
        )
    )

    answer, _ = run("Recommend a keto-friendly restaurant.")
    checks.append(
        (
            "keto evidence grounding",
            (
                "keto" in answer.lower()
                and (
                    "explicit evidence" in answer.lower()
                    or "low-carb" in answer.lower()
                    or "low carb" in answer.lower()
                )
            ),
        )
    )

    passed = 0
    for name, success in checks:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {name}")
        passed += int(success)

    print(f"\nValidation checks passed: {passed}/{len(checks)}")

    if passed != len(checks):
        raise AssertionError("One or more RAG validation checks failed.")


if __name__ == "__main__":
    main()
