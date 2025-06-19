#!/usr/bin/env python3
"""
evaluation_statistics.py

Analyse batch evaluation results and generate summary statistics,
visualisations and reports. Optionally filter out questions whose
RAG overall score is below a user-defined threshold.

Usage examples
--------------
python evaluation_statistics.py
python evaluation_statistics.py --file custom_results.json --threshold 3.0
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, normaltest, ttest_rel
from termcolor import colored


class EvaluationStatistics:
    """
    Analyzes and visualizes evaluation results from batch evaluation runs.
    """

    def __init__(self, results_file: str = "batch_evaluation_results.json"):
        self.results_file = results_file
        self.data = self._load_data()
        self.df = self._create_dataframe()
        self.metrics = [
            "relevance",
            "completeness",
            "correctness",
            "clarity",
            "usefulness",
        ]

    def _load_data(self) -> Dict[str, Any]:
        """Load and parse the evaluation results."""
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Initialize data structure
            data = {
                "metadata": raw_data.get("metadata", {}),
                "human": {metric: [] for metric in ["relevance", "completeness", "correctness", "clarity", "usefulness"]},
                "rag": {metric: [] for metric in ["relevance", "completeness", "correctness", "clarity", "usefulness"]},
                "human_overall": [],
                "rag_overall": [],
                "human_total_evaluated": 0,
                "rag_total_evaluated": 0,
                "questions": [],
                "raw_results": raw_data.get("results", []),
            }

            # Process each result
            for result in raw_data.get("results", []):
                if "evaluation" not in result or not result["evaluation"]:
                    continue

                evaluation = result["evaluation"]
                data["questions"].append(result.get("question", ""))

                # Extract human evaluation scores
                if "human_evaluation" in evaluation:
                    human_eval = evaluation["human_evaluation"]
                    human_scores = []
                    for metric in self.metrics:
                        if metric in human_eval and isinstance(human_eval[metric], (int, float)):
                            score = human_eval[metric]
                            data["human"][metric].append(score)
                            human_scores.append(score)

                    if human_scores:
                        data["human_overall"].append(mean(human_scores))
                        data["human_total_evaluated"] += 1

                # Extract RAG evaluation scores
                if "rag_evaluation" in evaluation:
                    rag_eval = evaluation["rag_evaluation"]
                    rag_scores = []
                    for metric in self.metrics:
                        if metric in rag_eval and isinstance(rag_eval[metric], (int, float)):
                            score = rag_eval[metric]
                            data["rag"][metric].append(score)
                            rag_scores.append(score)

                    if rag_scores:
                        data["rag_overall"].append(mean(rag_scores))
                        data["rag_total_evaluated"] += 1

            return data

        except FileNotFoundError:
            print(colored(f"Results file '{self.results_file}' not found", "red"))
            return {"metadata": {}, "human": {}, "rag": {}, "questions": [], "raw_results": []}
        except json.JSONDecodeError:
            print(colored(f"Invalid JSON in '{self.results_file}'", "red"))
            return {"metadata": {}, "human": {}, "rag": {}, "questions": [], "raw_results": []}
        except Exception as e:
            print(colored(f"Error loading data: {str(e)}", "red"))
            return {"metadata": {}, "human": {}, "rag": {}, "questions": [], "raw_results": []}

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from the loaded data."""
        rows = []
        for i, question in enumerate(self.data["questions"]):
            row = {"question": question, "question_id": i}

            # Add human scores
            for metric in self.metrics:
                if i < len(self.data["human"][metric]):
                    row[f"human_{metric}"] = self.data["human"][metric][i]

            # Add RAG scores
            for metric in self.metrics:
                if i < len(self.data["rag"][metric]):
                    row[f"rag_{metric}"] = self.data["rag"][metric][i]

            # Add overall scores
            if i < len(self.data["human_overall"]):
                row["human_overall"] = self.data["human_overall"][i]
            if i < len(self.data["rag_overall"]):
                row["rag_overall"] = self.data["rag_overall"][i]

            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        stats = {}

        # Overall statistics
        stats["total_evaluations"] = len(self.data["questions"])
        stats["human_evaluated"] = self.data["human_total_evaluated"]
        stats["rag_evaluated"] = self.data["rag_total_evaluated"]

        # Metric-wise statistics
        stats["metrics"] = {}
        for metric in self.metrics:
            human_scores = self.data["human"][metric]
            rag_scores = self.data["rag"][metric]

            stats["metrics"][metric] = {
                "human": {
                    "count": len(human_scores),
                    "mean": mean(human_scores) if human_scores else None,
                    "std": stdev(human_scores) if len(human_scores) > 1 else None,
                    "min": min(human_scores) if human_scores else None,
                    "max": max(human_scores) if human_scores else None,
                },
                "rag": {
                    "count": len(rag_scores),
                    "mean": mean(rag_scores) if rag_scores else None,
                    "std": stdev(rag_scores) if len(rag_scores) > 1 else None,
                    "min": min(rag_scores) if rag_scores else None,
                    "max": max(rag_scores) if rag_scores else None,
                },
            }

        # Overall score statistics
        if self.data["human_overall"]:
            stats["human_overall"] = {
                "mean": mean(self.data["human_overall"]),
                "std": stdev(self.data["human_overall"]) if len(self.data["human_overall"]) > 1 else None,
            }

        if self.data["rag_overall"]:
            stats["rag_overall"] = {
                "mean": mean(self.data["rag_overall"]),
                "std": stdev(self.data["rag_overall"]) if len(self.data["rag_overall"]) > 1 else None,
            }

        return stats

    # ------------------------------------------------------------------ #
    # Summary statistics printing
    # ------------------------------------------------------------------ #
    def print_summary_statistics(self) -> None:
        print("=" * 60)
        print("RAG EVALUATION SUMMARY STATISTICS")
        print("=" * 60)

        print("Total questions evaluated:")
        print(f"   Human answers: {self.data['human_total_evaluated']}")
        print(f"   RAG answers:   {self.data['rag_total_evaluated']}\n")

        # Overall averages
        ho = self.data.get("human_overall", [])
        ro = self.data.get("rag_overall", [])
        if ho:
            hm = mean(ho)
            hs = stdev(ho) if len(ho) > 1 else 0
            print("Overall Average Scores (1–5):")
            print(f"   Human: {hm:.2f} ± {hs:.2f}")
            if ro:
                rm = mean(ro)
                rs = stdev(ro) if len(ro) > 1 else 0
                diff = rm - hm
                tag = "(RAG better)" if diff > 0 else "(Human better)"
                print(f"   RAG:   {rm:.2f} ± {rs:.2f}")
                print(f"   Difference: {diff:+.2f} {tag}\n")

        print("Per-Metric Statistics:")
        print("-" * 45)
        print(f"{'Metric':<25}{'Human':<12}{'RAG':<12}{'Diff':<8}")
        print("-" * 45)
        for m in self.metrics:
            hscores = self.data["human"][m]
            rscores = self.data["rag"][m]
            hstr = rsstr = "No data"
            dstr = "N/A"
            if hscores:
                mm = mean(hscores)
                ss = stdev(hscores) if len(hscores) > 1 else 0
                hstr = f"{mm:.2f}±{ss:.2f}"
            if rscores:
                rm = mean(rscores)
                rs = stdev(rscores) if len(rscores) > 1 else 0
                rsstr = f"{rm:.2f}±{rs:.2f}"
                if hscores:
                    dstr = f"{(rm - mm):+.2f}"
            print(f"{m.replace('_', ' ').title():<25}{hstr:<12}{rsstr:<12}{dstr:<8}")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Visualization methods
    # ------------------------------------------------------------------ #
    def create_radar_chart(self) -> None:
        """Create a radar chart comparing human vs RAG scores across metrics."""
        human_means = [
            mean(self.data["human"][m]) if self.data["human"][m] else 0
            for m in self.metrics
        ]
        rag_means = [
            mean(self.data["rag"][m]) if self.data["rag"][m] else 0
            for m in self.metrics
        ]

        if not any(human_means) and not any(rag_means):
            print("Insufficient data for radar chart")
            return

        angles = np.linspace(0, 2 * np.pi, len(self.metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        human_means.append(human_means[0])  # Complete the circle
        rag_means.append(rag_means[0])  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        ax.plot(angles, human_means, "o-", linewidth=2, label="Human", color="blue")
        ax.fill(angles, human_means, alpha=0.25, color="blue")
        ax.plot(angles, rag_means, "o-", linewidth=2, label="RAG", color="red")
        ax.fill(angles, rag_means, alpha=0.25, color="red")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in self.metrics])
        ax.set_ylim(0, 5)
        ax.set_title("Human vs RAG Evaluation Scores", size=16, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def _create_mean_comparison_chart(self):
        hm = {
            m: mean(self.data["human"][m])
            for m in self.metrics
            if self.data["human"][m]
        }
        rm = {m: mean(self.data["rag"][m]) for m in self.metrics if self.data["rag"][m]}
        if not hm or not rm:
            return print("Insufficient data for bar chart")
        df = pd.DataFrame({"Human": pd.Series(hm), "RAG": pd.Series(rm)}).fillna(0)
        ax = df.plot(kind="bar", figsize=(12, 6), width=0.8)
        plt.title("Mean Scores Comparison", fontweight="bold")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", label_type="edge")
        plt.tight_layout()
        plt.show()

    def _create_overall_distribution(self):
        ho, ro = self.data["human_overall"], self.data["rag_overall"]
        if not ho or not ro:
            return print("Insufficient data for distribution")
        plt.figure(figsize=(10, 6))
        bins = np.linspace(1, 5, 20)
        plt.hist(ho, bins=bins, alpha=0.7, label="Human", density=True)
        plt.hist(ro, bins=bins, alpha=0.7, label="RAG", density=True)
        plt.axvline(mean(ho), linestyle="--", label=f"Human Mean: {mean(ho):.2f}")
        plt.axvline(mean(ro), linestyle="--", label=f"RAG Mean: {mean(ro):.2f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("Overall Score Distribution", fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _create_boxplots(self):
        for m in self.metrics:
            h = self.df[f"human_{m}"].dropna()
            r = self.df[f"rag_{m}"].dropna()
            if h.empty and r.empty:
                continue
            plt.figure(figsize=(8, 6))
            df = pd.DataFrame(
                {
                    "Score": pd.concat([h, r]),
                    "Type": ["Human"] * len(h) + ["RAG"] * len(r),
                }
            )
            sns.boxplot(data=df, x="Type", y="Score", showmeans=True)
            plt.title(f"{m.replace('_', ' ').title()} Distribution", fontweight="bold")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _create_difference_analysis(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, m in enumerate(self.metrics):
            dfm = self.df.dropna(subset=[f"human_{m}", f"rag_{m}"])
            if dfm.empty:
                axes[i].text(0.5, 0.5, "No paired data", ha="center", va="center")
                axes[i].set_title(m.replace("_", " ").title())
                continue
            diff = dfm[f"rag_{m}"] - dfm[f"human_{m}"]
            axes[i].hist(diff, bins=11, alpha=0.7, edgecolor="black")
            axes[i].axvline(0, linestyle="--")
            axes[i].axvline(diff.mean(), label=f"Mean: {diff.mean():.2f}")
            axes[i].set_title(f"{m.replace('_', ' ').title()} Δ")
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        plt.suptitle("Score Differences (RAG - Human)", fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _create_correlation_analysis(self):
        cols = [f"rag_{m}" for m in self.metrics]
        dfr = self.df[cols].dropna()
        if dfr.empty:
            return
        dfr.columns = [m.replace("_", " ").title() for m in self.metrics]
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones((len(cols), len(cols)), dtype=bool))
        sns.heatmap(dfr.corr(), annot=True, mask=mask, fmt=".2f")
        plt.title("RAG Metric Correlations", fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _perform_statistical_tests(self):
        print("\n" + "=" * 50 + "\nSTATISTICAL SIGNIFICANCE TESTS\n" + "=" * 50)
        res = []
        for m in self.metrics:
            dfm = self.df.dropna(subset=[f"human_{m}", f"rag_{m}"])
            if len(dfm) < 3:
                continue
            hs, rs = dfm[f"human_{m}"], dfm[f"rag_{m}"]
            t_stat, t_p = ttest_rel(rs, hs)
            u_stat, u_p = mannwhitneyu(rs, hs)
            diff = rs - hs
            _, norm_p = normaltest(diff)
            sd = np.sqrt((np.var(hs, ddof=1) + np.var(rs, ddof=1)) / 2)
            d = diff.mean() / sd if sd > 0 else 0
            eff = (
                (abs(d) < 0.2 and "Negligible")
                or (abs(d) < 0.5 and "Small")
                or (abs(d) < 0.8 and "Medium")
                or "Large"
            )
            prat = "✓" if abs(diff.mean()) >= 0.5 else "✗"
            sig = "✓" if t_p < 0.05 else "✗"
            res.append((m.title(), len(dfm), diff.mean(), d, eff, sig, prat, t_p))
        if res:
            print(
                f"{'Metric':<15}{'N':<5}{'Mean Δ':<10}{'Eff':<10}{'Stat':<6}{'Prac':<6}{'p-val':<8}"
            )
            print("-" * 60)
            for r in res:
                print(
                    f"{r[0]:<15}{r[1]:<5}{r[2]:<10.2f}{r[4]:<10}{r[5]:<6}{r[6]:<6}{r[7]:<8.3f}"
                )

    def _create_score_distributions(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, m in enumerate(self.metrics):
            data = []
            for grp in ["human", "rag"]:
                s = self.df[f"{grp}_{m}"].dropna()
                if not s.empty:
                    vc = s.value_counts().sort_index()
                    for sc, ct in vc.items():
                        data.append(
                            {"Group": grp.title(), "Score": int(sc), "Count": ct}
                        )
            if data:
                sns.barplot(
                    data=pd.DataFrame(data),
                    x="Score",
                    y="Count",
                    hue="Group",
                    ax=axes[i],
                )
            else:
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].set_title(m.replace("_", " ").title())
            axes[i].set_xlabel("Score")
            axes[i].set_ylabel("Count")
            axes[i].set_xticks(range(1, 6))
        plt.suptitle("Score Distribution by Metric", fontweight="bold")
        plt.tight_layout()
        plt.show()

    def create_comprehensive_report(self, filename: str = "evaluation_report.html") -> None:
        """Generate a comprehensive HTML report with all visualizations."""
        try:
            self.print_summary_statistics()
            self.create_radar_chart()
            self._create_mean_comparison_chart()
            self._create_overall_distribution()
            self._create_boxplots()
            self._create_difference_analysis()
            self._create_correlation_analysis()
            self._perform_statistical_tests()
            self._create_score_distributions()
            print(f"Report saved to {filename}")
        except Exception as e:
            print(colored(f"Error creating report: {str(e)}", "red"))

    def analyze(self):
        """Run a complete analysis with all available methods."""
        try:
            print(colored("Generating detailed analysis...", "blue"))
            self.print_summary_statistics()
            self.create_radar_chart()
            print("Analysis complete!")
        except KeyboardInterrupt:
            print(colored("Analysis interrupted by user", "yellow"))
        except Exception as e:
            print(colored(f"Unexpected error: {e}", "red"))


def main():
    parser = argparse.ArgumentParser(description="Analyse batch evaluation results.")
    parser.add_argument(
        "--file",
        "-f",
        default="batch_evaluation_results.json",
        help="Path to JSON file.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Min RAG overall to include.",
    )
    args = parser.parse_args()
    try:
        stats = EvaluationStatistics(results_file=args.file)
        stats.analyze()
    except FileNotFoundError as e:
        print(colored(f"Error: {e}", "red"))
        print("Run batch evaluation first to generate data.")
    except Exception as e:
        print(colored(f"Unexpected error: {e}", "red"))


if __name__ == "__main__":
    main()

