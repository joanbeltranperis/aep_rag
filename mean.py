import json
from statistics import mean


def calculate_separated_means(data):
    metrics = [
        "correctness",
        "completeness",
        "relevance",
        "clarity_and_fluency",
        "alignment_with_intent",
    ]

    def init_metric_dict():
        return {metric: [] for metric in metrics}

    scores = {
        "human": init_metric_dict(),
        "rag": init_metric_dict(),
        "human_overall": [],
        "rag_overall": [],
        "human_total_evaluated": 0,
        "rag_total_evaluated": 0,
    }

    for question in data.get("questions", []):
        for key in ["human", "rag"]:
            answer = question.get(f"{key}_answer")
            if not answer or not answer.get("evaluation"):
                continue
            eval_data = answer["evaluation"]

            metric_scores = []
            for metric in metrics:
                score = eval_data.get(metric)
                if score is not None:
                    scores[key][metric].append(score)
                    metric_scores.append(score)

            if metric_scores:
                scores[f"{key}_overall"].append(mean(metric_scores))
                scores[f"{key}_total_evaluated"] += 1

    def compute_means(score_dict):
        return {
            metric: round(mean(values), 2)
            for metric, values in score_dict.items()
            if values
        }

    return {
        "human_means": compute_means(scores["human"]),
        "rag_means": compute_means(scores["rag"]),
        "human_overall_mean": round(mean(scores["human_overall"]), 2)
        if scores["human_overall"]
        else None,
        "rag_overall_mean": round(mean(scores["rag_overall"]), 2)
        if scores["rag_overall"]
        else None,
        "human_total_evaluated": scores["human_total_evaluated"],
        "rag_total_evaluated": scores["rag_total_evaluated"],
    }


# Example usage
if __name__ == "__main__":
    with open("questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    results = calculate_separated_means(data)
    print(json.dumps(results, indent=2, ensure_ascii=False))
