import json
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

# Define the evaluation metrics.
metrics = [
    "correctness",
    "completeness",
    "relevance",
    "clarity_and_fluency",
    "alignment_with_intent",
]


def init_metric_dict():
    """Initialize a dictionary for each metric."""
    return {metric: [] for metric in metrics}


# Collect scores and per-question data.
scores = {
    "human": init_metric_dict(),
    "rag": init_metric_dict(),
    "human_overall": [],
    "rag_overall": [],
    "human_total_evaluated": 0,
    "rag_total_evaluated": 0,
    "per_question": [],
}

# Load the evaluation data.
with open("questions_evaluation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process every question.
for question in data.get("questions", []):
    question_entry = {"id": question.get("id")}
    for key in ["human", "rag"]:
        answer = question.get(f"{key}_answer")
        if not answer or not answer.get("evaluation"):
            continue
        eval_data = answer["evaluation"]
        metric_scores = []
        for metric in metrics:
            score = eval_data.get(metric)
            if score is not None:
                # Append score to global lists.
                scores[key][metric].append(score)
                # Record this score for the current question.
                question_entry[f"{key}_{metric}"] = score
                metric_scores.append(score)
        if metric_scores:
            current_mean = mean(metric_scores)
            scores[f"{key}_overall"].append(current_mean)
            scores[f"{key}_total_evaluated"] += 1
            question_entry[f"{key}_overall"] = current_mean
    scores["per_question"].append(question_entry)

# Build a DataFrame with all questions.
df = pd.DataFrame(scores["per_question"])

#########################################################
# 1. Boxplots per Metric (Human vs. RAG)
#########################################################
for metric in metrics:
    # Use only rows that have a value for this metric.
    human_scores = df[f"human_{metric}"].dropna()
    rag_scores = df[f"rag_{metric}"].dropna()

    if human_scores.empty and rag_scores.empty:
        print(f"No data for metric: {metric}")
        continue

    plt.figure(figsize=(6, 4))
    # Prepare a DataFrame for seaborn.
    bp_df = pd.DataFrame(
        {
            "Score": pd.concat([human_scores, rag_scores]),
            "Group": ["Human"] * len(human_scores) + ["RAG"] * len(rag_scores),
        }
    )
    sns.boxplot(x="Group", y="Score", data=bp_df)
    plt.title(f"Boxplot: {metric.replace('_', ' ').title()}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()


#########################################################
# 2. Radar Chart (Spider Chart) of Mean Scores
#########################################################
def radar_chart(human_means, rag_means):
    labels = np.array(metrics)
    # Retrieve mean scores in the order of the metrics.
    stats_h = np.array([human_means[m] for m in labels])
    stats_r = np.array([rag_means[m] for m in labels])

    # Calculate angles for radar chart.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # Complete the loop.
    stats_h = np.concatenate((stats_h, [stats_h[0]]))
    stats_r = np.concatenate((stats_r, [stats_r[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_h, label="Human", linewidth=2)
    ax.fill(angles, stats_h, alpha=0.25)
    ax.plot(angles, stats_r, label="RAG", linewidth=2)
    ax.fill(angles, stats_r, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in labels])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    plt.title("Radar Chart of Mean Scores")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# Calculate mean scores per metric from the global scores.
human_means = {
    m: round(mean(scores["human"][m]), 2) for m in metrics if scores["human"][m]
}
rag_means = {m: round(mean(scores["rag"][m]), 2) for m in metrics if scores["rag"][m]}
radar_chart(human_means, rag_means)

#########################################################
# 3. Bar Chart of Mean Scores for Each Metric
#########################################################
mean_df = pd.DataFrame({"Human": pd.Series(human_means), "RAG": pd.Series(rag_means)})
mean_df.plot(kind="bar", figsize=(8, 5))
plt.title("Mean Scores per Metric")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#########################################################
# 4. Histograms for Score Differences (RAG - Human) per Metric
#########################################################
for metric in metrics:
    # For paired comparisons, select only questions having both scores.
    df_metric = df.dropna(subset=[f"human_{metric}", f"rag_{metric}"])
    if df_metric.empty:
        print(f"No paired data for metric: {metric}")
        continue
    diff = df_metric[f"rag_{metric}"] - df_metric[f"human_{metric}"]
    plt.figure(figsize=(6, 4))
    sns.histplot(diff, kde=True, bins=10)
    plt.axvline(0, color="red", linestyle="--", label="No Difference")
    plt.title(f"Difference (RAG - Human) in {metric.replace('_', ' ').title()}")
    plt.xlabel("Score Difference")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

#########################################################
# 5. Correlation Matrix of RAG Scores Across Metrics
#########################################################
# Select rows that have all RAG metric evaluations.
rag_cols = [f"rag_{m}" for m in metrics]
df_rag = df[rag_cols].dropna()
if df_rag.empty:
    print("No complete RAG data available for correlation matrix.")
else:
    # Rename columns for clarity.
    df_rag.columns = metrics
    corr_matrix = df_rag.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (RAG Scores)")
    plt.tight_layout()
    plt.show()

#########################################################
# 6. Paired t-test per Metric (Human vs. RAG)
#########################################################
t_test_results = {}
for metric in metrics:
    df_metric = df.dropna(subset=[f"human_{metric}", f"rag_{metric}"])
    if df_metric.empty:
        print(f"Skipping t-test for {metric} due to insufficient paired data.")
        continue
    t_result = ttest_rel(df_metric[f"rag_{metric}"], df_metric[f"human_{metric}"])
    t_test_results[metric] = {
        "t-statistic": round(t_result.statistic, 4),
        "p-value": round(t_result.pvalue, 4),
    }

# Show paired t-test results.
t_test_summary = pd.DataFrame.from_dict(t_test_results, orient="index")
try:
    import ace_tools  # if available in your environment

    ace_tools.display_dataframe_to_user(
        name="T-Test Results per Metric", dataframe=t_test_summary
    )
except ImportError:
    print("Paired t-test results per metric:")
    print(t_test_summary)

#########################################################
# Optional: Overall Summary Statistics
#########################################################
overall_summary = {
    "human_means": human_means,
    "rag_means": rag_means,
    "human_overall_mean": round(mean(scores["human_overall"]), 2)
    if scores["human_overall"]
    else None,
    "rag_overall_mean": round(mean(scores["rag_overall"]), 2)
    if scores["rag_overall"]
    else None,
    "human_total_evaluated": scores["human_total_evaluated"],
    "rag_total_evaluated": scores["rag_total_evaluated"],
}

print("Overall Summary Statistics:")
print(json.dumps(overall_summary, indent=2, ensure_ascii=False))

#########################################################
# 7. Distribution of Exact Scores per Metric (Human vs RAG)
#########################################################
for metric in metrics:
    plt.figure(figsize=(8, 5))
    # Create a DataFrame with score distributions
    dist_data = []
    for group in ["human", "rag"]:
        score_series = df[f"{group}_{metric}"].dropna().astype(int)
        score_counts = score_series.value_counts().sort_index()
        for score, count in score_counts.items():
            dist_data.append(
                {"Group": group.capitalize(), "Score": score, "Count": count}
            )

    dist_df = pd.DataFrame(dist_data)

    sns.barplot(data=dist_df, x="Score", y="Count", hue="Group")
    plt.title(f"Distribution of Exact Scores: {metric.replace('_', ' ').title()}")
    plt.xticks([0, 1, 2, 3, 4], labels=[1, 2, 3, 4, 5])
    plt.ylabel("Number of Responses")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.show()
