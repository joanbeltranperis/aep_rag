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

# Ignore harmless warnings (e.g. scipy RuntimeWarnings for small samples)
import warnings
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, normaltest, ttest_rel

warnings.filterwarnings("ignore", category=RuntimeWarning)


class EvaluationStatistics:
    """Compute statistics and visualisations for batch evaluation results."""

    def __init__(
        self,
        json_file: str = "batch_evaluation_results.json",
        rag_threshold: Optional[float] = None,
    ) -> None:
        """Create a new evaluator.

        Parameters
        ----------
        json_file : str
            Path to the JSON produced by the batch evaluation pipeline.
        rag_threshold : float | None
            If provided, skip any question whose average **RAG** score
            (mean of the 5 metrics) is **strictly lower** than this value.
        """
        self.json_file = Path(json_file)
        self.rag_threshold = rag_threshold

        self.metrics: List[str] = [
            "correctness",
            "completeness",
            "relevance",
            "clarity_and_fluency",
            "alignment_with_intent",
        ]

        self.data: Dict = {}
        self.df: pd.DataFrame = pd.DataFrame()

        # Configure default plotting style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            plt.style.use("default")
        sns.set_palette("husl")

    # ------------------------------------------------------------------ #
    # Loading & preprocessing
    # ------------------------------------------------------------------ #
    def load_and_process_data(self) -> None:
        """Read the evaluation JSON and turn it into a tidy DataFrame."""
        if not self.json_file.exists():
            raise FileNotFoundError(f"Evaluation file '{self.json_file}' not found")

        with open(self.json_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = self._process_raw_data(raw_data)
        self.df = pd.DataFrame(self.data["per_question"])

    def _process_raw_data(self, raw_data: dict) -> dict:
        """Convert raw JSON into structured statistics (with optional filter)."""
        scores = {
            "human": {m: [] for m in self.metrics},
            "rag": {m: [] for m in self.metrics},
            "human_overall": [],
            "rag_overall": [],
            "human_total_evaluated": 0,
            "rag_total_evaluated": 0,
            "per_question": [],
        }

        for result in raw_data.get("results", []):
            q_entry: Dict = {"id": result.get("original_index")}
            evaluation = result.get("evaluation", {})
            if not evaluation:
                continue

            # Collect metric scores per answer type
            local_human: Dict[str, float] = {}
            local_rag: Dict[str, float] = {}
            human_scores_list: List[float] = []
            rag_scores_list: List[float] = []

            for answer_key, eval_key in [
                ("human", "human_evaluation"),
                ("rag", "rag_evaluation"),
            ]:
                eval_data = evaluation.get(eval_key, {})
                if not eval_data:
                    continue

                for metric in self.metrics:
                    score = eval_data.get(metric)
                    if score is None:
                        continue
                    q_entry[f"{answer_key}_{metric}"] = score
                    if answer_key == "human":
                        local_human[metric] = score
                        human_scores_list.append(score)
                    else:
                        local_rag[metric] = score
                        rag_scores_list.append(score)

            # Compute overall means
            if human_scores_list:
                q_entry["human_overall"] = mean(human_scores_list)
            if rag_scores_list:
                q_entry["rag_overall"] = mean(rag_scores_list)

            # Apply rag threshold filter
            if (
                self.rag_threshold is not None
                and q_entry.get("rag_overall", 0) < self.rag_threshold
            ):
                continue

            # Append scores to aggregate
            for m, v in local_human.items():
                scores["human"][m].append(v)
            for m, v in local_rag.items():
                scores["rag"][m].append(v)

            if "human_overall" in q_entry:
                scores["human_overall"].append(q_entry["human_overall"])
                scores["human_total_evaluated"] += 1
            if "rag_overall" in q_entry:
                scores["rag_overall"].append(q_entry["rag_overall"])
                scores["rag_total_evaluated"] += 1

            scores["per_question"].append(q_entry)

        return scores

    # ------------------------------------------------------------------ #
    # Summary statistics printing
    # ------------------------------------------------------------------ #
    def print_summary_statistics(self) -> None:
        print("=" * 60)
        print("RAG EVALUATION SUMMARY STATISTICS")
        print("=" * 60)

        print("📊 Total questions evaluated:")
        print(f"   Human answers: {self.data['human_total_evaluated']}")
        print(f"   RAG answers:   {self.data['rag_total_evaluated']}\n")

        # Overall averages
        ho = self.data.get("human_overall", [])
        ro = self.data.get("rag_overall", [])
        if ho:
            hm = mean(ho)
            hs = stdev(ho) if len(ho) > 1 else 0
            print("🎯 Overall Average Scores (1–5):")
            print(f"   Human: {hm:.2f} ± {hs:.2f}")
            if ro:
                rm = mean(ro)
                rs = stdev(ro) if len(ro) > 1 else 0
                diff = rm - hm
                tag = "(RAG better)" if diff > 0 else "(Human better)"
                print(f"   RAG:   {rm:.2f} ± {rs:.2f}")
                print(f"   Difference: {diff:+.2f} {tag}\n")

        print("📈 Per-Metric Statistics:")
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
    def generate_summary_plots(self) -> None:
        self._create_radar_chart()
        self._create_mean_comparison_chart()
        self._create_overall_distribution()

    def generate_detailed_analysis(self) -> None:
        self._create_boxplots()
        self._create_difference_analysis()
        self._create_correlation_analysis()
        self._perform_statistical_tests()
        self._create_score_distributions()

    def _create_radar_chart(self):
        human_means = {
            m: mean(self.data["human"][m])
            for m in self.metrics
            if self.data["human"][m]
        }
        rag_means = {
            m: mean(self.data["rag"][m]) for m in self.metrics if self.data["rag"][m]
        }
        if not human_means or not rag_means:
            print("⚠️ Insufficient data for radar chart")
            return
        labels = np.array(list(human_means.keys()))
        hv = np.array(list(human_means.values()))
        rv = np.array([rag_means[m] for m in labels])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        hv = np.concatenate((hv, [hv[0]]))
        rv = np.concatenate((rv, [rv[0]]))
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 8))
        ax.plot(angles, hv, "o-", lw=2, label="Human")
        ax.fill(angles, hv, alpha=0.25)
        ax.plot(angles, rv, "o-", lw=2, label="RAG")
        ax.fill(angles, rv, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([l.title() for l in labels])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 5)
        ax.grid(True)
        plt.title("Performance Comparison: Human vs RAG", pad=20, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.3, 1.0))
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
            return print("⚠️ Insufficient data for bar chart")
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
            return print("⚠️ Insufficient data for distribution")
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

    def save_summary_report(self, filename: str = "evaluation_report.txt") -> None:
        """Save a text summary report to disk."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("RAG EVALUATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Total questions:\n  Human: {self.data['human_total_evaluated']}\n  RAG: {self.data['rag_total_evaluated']}\n\n"
            )
            f.write("Per-metric:\n")
            for m in self.metrics:
                hs = self.data["human"][m]
                rs = self.data["rag"][m]
                if hs and rs:
                    hm, hsd = mean(hs), stdev(hs) if len(hs) > 1 else 0
                    rm, rsd = mean(rs), stdev(rs) if len(rs) > 1 else 0
                    diff = rm - hm
                    f.write(
                        f"  {m.title()}: Human {hm:.2f}±{hsd:.2f}, RAG {rm:.2f}±{rsd:.2f}, Δ{diff:+.2f}\n"
                    )
        print(f"📄 Report saved to {filename}")


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
        stats = EvaluationStatistics(json_file=args.file, rag_threshold=2)
        stats.load_and_process_data()
        stats.print_summary_statistics()
        print("\n🎨 Generating summary visualizations...")
        stats.generate_summary_plots()
        print("\n📊 Generating detailed analysis...")
        stats.generate_detailed_analysis()
        stats.save_summary_report()
        print("\n✅ Analysis complete!")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Run batch evaluation first to generate data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

