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

    def __init__(self, results_file: str = "batch_evaluation_results.json", threshold: float = None):
        self.results_file = results_file
        self.threshold = threshold
        # Define metrics before loading data (which needs them)
        self.metrics = [
            "correctness",
            "completeness", 
            "relevance",
            "clarity_and_fluency",
            "alignment_with_intent",
        ]
        self.data = self._load_data()
        self.df = self._create_dataframe()

    def _load_data(self) -> Dict[str, Any]:
        """Load and parse the evaluation results."""
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Initialize data structure
            data = {
                "metadata": raw_data.get("metadata", {}),
                "human": {metric: [] for metric in self.metrics},
                "rag": {metric: [] for metric in self.metrics},
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
                
                # First, check if we need to apply threshold filtering
                include_result = True
                if self.threshold is not None and "rag_evaluation" in evaluation:
                    rag_eval = evaluation["rag_evaluation"]
                    rag_scores = []
                    for metric in self.metrics:
                        if metric in rag_eval and isinstance(rag_eval[metric], (int, float)):
                            rag_scores.append(rag_eval[metric])
                    
                    if rag_scores:
                        rag_overall_score = mean(rag_scores)
                        include_result = rag_overall_score >= self.threshold
                
                # Only include this result if it passes the threshold filter
                if include_result:
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

            # Print threshold filtering info if applied
            if self.threshold is not None:
                total_results = len(raw_data.get("results", []))
                filtered_results = len(data["questions"])
                print(colored(f"Threshold filter applied: {self.threshold}", "blue"))
                print(colored(f"Results: {filtered_results}/{total_results} included (RAG overall ≥ {self.threshold})", "blue"))

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

    def create_scatter_plots(self):
        """Create scatter plots showing human vs RAG scores for each metric."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics):
            human_scores = []
            rag_scores = []
            
            # Get paired scores
            for j in range(len(self.data["questions"])):
                if (j < len(self.data["human"][metric]) and 
                    j < len(self.data["rag"][metric])):
                    human_scores.append(self.data["human"][metric][j])
                    rag_scores.append(self.data["rag"][metric][j])
            
            if human_scores and rag_scores:
                axes[i].scatter(human_scores, rag_scores, alpha=0.6, s=50)
                axes[i].plot([1, 5], [1, 5], 'r--', alpha=0.8, label='Perfect Agreement')
                axes[i].set_xlabel('Human Score')
                axes[i].set_ylabel('RAG Score')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                axes[i].set_xlim(0.5, 5.5)
                axes[i].set_ylim(0.5, 5.5)
                
                # Add correlation coefficient
                corr = np.corrcoef(human_scores, rag_scores)[0, 1]
                axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                axes[i].text(0.5, 0.5, 'No paired data', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
        
        plt.suptitle('Human vs RAG Score Correlations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def create_violin_plots(self):
        """Create violin plots showing score distributions for each metric."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics):
            data_for_violin = []
            labels = []
            
            if self.data["human"][metric]:
                data_for_violin.append(self.data["human"][metric])
                labels.append('Human')
                
            if self.data["rag"][metric]:
                data_for_violin.append(self.data["rag"][metric])
                labels.append('RAG')
            
            if data_for_violin:
                parts = axes[i].violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                                         showmeans=True, showmedians=True)
                axes[i].set_xticks(range(len(labels)))
                axes[i].set_xticklabels(labels)
                axes[i].set_ylabel('Score')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0.5, 5.5)
                
                # Color the violins differently
                for j, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(['lightblue', 'lightcoral'][j] if j < 2 else 'lightgray')
                    pc.set_alpha(0.7)
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
        
        plt.suptitle('Score Distribution Shapes (Violin Plots)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def create_agreement_analysis(self):
        """Analyze agreement between human and RAG scores."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall agreement scatter
        human_overall = self.data.get("human_overall", [])
        rag_overall = self.data.get("rag_overall", [])
        
        if human_overall and rag_overall:
            min_len = min(len(human_overall), len(rag_overall))
            h_scores = human_overall[:min_len]
            r_scores = rag_overall[:min_len]
            
            ax1.scatter(h_scores, r_scores, alpha=0.6, s=60)
            ax1.plot([1, 5], [1, 5], 'r--', alpha=0.8, label='Perfect Agreement')
            ax1.set_xlabel('Human Overall Score')
            ax1.set_ylabel('RAG Overall Score')
            ax1.set_title('Overall Score Agreement')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add correlation
            corr = np.corrcoef(h_scores, r_scores)[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Agreement levels
            differences = [abs(h - r) for h, r in zip(h_scores, r_scores)]
            perfect_agreement = sum(1 for d in differences if d == 0)
            close_agreement = sum(1 for d in differences if 0 < d <= 0.5)
            moderate_agreement = sum(1 for d in differences if 0.5 < d <= 1.0)
            poor_agreement = sum(1 for d in differences if d > 1.0)
            
            # Agreement pie chart
            agreement_data = [perfect_agreement, close_agreement, moderate_agreement, poor_agreement]
            agreement_labels = ['Perfect (0)', 'Close (≤0.5)', 'Moderate (≤1.0)', 'Poor (>1.0)']
            colors = ['green', 'lightgreen', 'orange', 'red']
            
            ax2.pie(agreement_data, labels=agreement_labels, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Agreement Levels Distribution')
            
            # Difference histogram
            ax3.hist(differences, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(differences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(differences):.2f}')
            ax3.set_xlabel('Absolute Difference |Human - RAG|')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Score Difference Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Score range analysis
            score_ranges = ['1.0-2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0']
            human_range_counts = [sum(1 for s in h_scores if i <= s < i+1) for i in range(1, 5)]
            rag_range_counts = [sum(1 for s in r_scores if i <= s < i+1) for i in range(1, 5)]
            
            x = np.arange(len(score_ranges))
            width = 0.35
            
            ax4.bar(x - width/2, human_range_counts, width, label='Human', alpha=0.7)
            ax4.bar(x + width/2, rag_range_counts, width, label='RAG', alpha=0.7)
            ax4.set_xlabel('Score Range')
            ax4.set_ylabel('Count')
            ax4.set_title('Score Range Distribution')
            ax4.set_xticks(x)
            ax4.set_xticklabels(score_ranges)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_performance_heatmap(self):
        """Create a heatmap showing performance across questions and metrics."""
        if not self.data["questions"]:
            print("No data available for performance heatmap")
            return
            
        # Create matrices for human and RAG scores
        num_questions = len(self.data["questions"])
        human_matrix = np.full((num_questions, len(self.metrics)), np.nan)
        rag_matrix = np.full((num_questions, len(self.metrics)), np.nan)
        
        for i, metric in enumerate(self.metrics):
            for j in range(min(num_questions, len(self.data["human"][metric]))):
                human_matrix[j, i] = self.data["human"][metric][j]
            for j in range(min(num_questions, len(self.data["rag"][metric]))):
                rag_matrix[j, i] = self.data["rag"][metric][j]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        
        # Human performance heatmap
        sns.heatmap(human_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=[m.replace('_', ' ').title() for m in self.metrics],
                   yticklabels=[f'Q{i+1}' for i in range(num_questions)],
                   vmin=1, vmax=5, ax=ax1)
        ax1.set_title('Human Scores by Question')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Questions')
        
        # RAG performance heatmap
        sns.heatmap(rag_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=[m.replace('_', ' ').title() for m in self.metrics],
                   yticklabels=[f'Q{i+1}' for i in range(num_questions)],
                   vmin=1, vmax=5, ax=ax2)
        ax2.set_title('RAG Scores by Question')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Questions')
        
        # Difference heatmap (RAG - Human)
        diff_matrix = rag_matrix - human_matrix
        sns.heatmap(diff_matrix, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                   xticklabels=[m.replace('_', ' ').title() for m in self.metrics],
                   yticklabels=[f'Q{i+1}' for i in range(num_questions)],
                   ax=ax3)
        ax3.set_title('Difference (RAG - Human)')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Questions')
        
        plt.tight_layout()
        plt.show()

    def create_metric_importance_analysis(self):
        """Analyze which metrics show the biggest differences and variations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        metric_stats = {}
        for metric in self.metrics:
            human_scores = self.data["human"][metric]
            rag_scores = self.data["rag"][metric]
            
            if human_scores and rag_scores:
                min_len = min(len(human_scores), len(rag_scores))
                h_scores = human_scores[:min_len]
                r_scores = rag_scores[:min_len]
                
                differences = [r - h for h, r in zip(h_scores, r_scores)]
                
                metric_stats[metric] = {
                    'mean_diff': np.mean(differences),
                    'abs_mean_diff': np.mean([abs(d) for d in differences]),
                    'std_diff': np.std(differences),
                    'human_std': np.std(h_scores),
                    'rag_std': np.std(r_scores),
                    'correlation': np.corrcoef(h_scores, r_scores)[0, 1] if len(h_scores) > 1 else 0
                }
        
        if not metric_stats:
            print("No paired data available for metric importance analysis")
            return
        
        metrics_clean = [m.replace('_', ' ').title() for m in metric_stats.keys()]
        
        # Mean differences (RAG - Human)
        mean_diffs = [metric_stats[m]['mean_diff'] for m in metric_stats.keys()]
        colors = ['red' if x < 0 else 'green' for x in mean_diffs]
        ax1.bar(metrics_clean, mean_diffs, color=colors, alpha=0.7)
        ax1.set_title('Mean Score Differences (RAG - Human)')
        ax1.set_ylabel('Mean Difference')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Absolute mean differences (disagreement magnitude)
        abs_mean_diffs = [metric_stats[m]['abs_mean_diff'] for m in metric_stats.keys()]
        ax2.bar(metrics_clean, abs_mean_diffs, color='orange', alpha=0.7)
        ax2.set_title('Mean Absolute Differences (Disagreement)')
        ax2.set_ylabel('Mean Absolute Difference')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Standard deviations (variability)
        human_stds = [metric_stats[m]['human_std'] for m in metric_stats.keys()]
        rag_stds = [metric_stats[m]['rag_std'] for m in metric_stats.keys()]
        
        x = np.arange(len(metrics_clean))
        width = 0.35
        ax3.bar(x - width/2, human_stds, width, label='Human', alpha=0.7)
        ax3.bar(x + width/2, rag_stds, width, label='RAG', alpha=0.7)
        ax3.set_title('Score Variability by Metric')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_clean, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Correlations
        correlations = [metric_stats[m]['correlation'] for m in metric_stats.keys()]
        ax4.bar(metrics_clean, correlations, color='purple', alpha=0.7)
        ax4.set_title('Human-RAG Correlations by Metric')
        ax4.set_ylabel('Correlation Coefficient')
        ax4.set_ylim(-1, 1)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_score_evolution_analysis(self):
        """Analyze how scores evolve across questions (if there's any temporal ordering)."""
        if not self.data["questions"]:
            print("No data available for score evolution analysis")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        human_overall = self.data.get("human_overall", [])
        rag_overall = self.data.get("rag_overall", [])
        
        if human_overall and rag_overall:
            question_indices = list(range(1, len(human_overall) + 1))
            
            # Overall score evolution
            ax1.plot(question_indices, human_overall, 'o-', label='Human', alpha=0.7)
            ax1.plot(question_indices, rag_overall, 's-', label='RAG', alpha=0.7)
            ax1.set_xlabel('Question Number')
            ax1.set_ylabel('Overall Score')
            ax1.set_title('Score Evolution Across Questions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Moving averages
            window_size = min(5, len(human_overall) // 2) if len(human_overall) >= 3 else 1
            if window_size > 1:
                human_ma = pd.Series(human_overall).rolling(window=window_size).mean()
                rag_ma = pd.Series(rag_overall).rolling(window=window_size).mean()
                
                ax2.plot(question_indices, human_ma, '-', label=f'Human (MA-{window_size})', linewidth=2)
                ax2.plot(question_indices, rag_ma, '-', label=f'RAG (MA-{window_size})', linewidth=2)
                ax2.set_xlabel('Question Number')
                ax2.set_ylabel('Overall Score (Moving Average)')
                ax2.set_title('Score Trends (Moving Average)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Score differences over time
            differences = [r - h for h, r in zip(human_overall, rag_overall)]
            ax3.plot(question_indices, differences, 'o-', color='red', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Question Number')
            ax3.set_ylabel('Score Difference (RAG - Human)')
            ax3.set_title('Performance Gap Over Questions')
            ax3.grid(True, alpha=0.3)
            
            # Cumulative performance
            human_cumsum = np.cumsum(human_overall) / np.arange(1, len(human_overall) + 1)
            rag_cumsum = np.cumsum(rag_overall) / np.arange(1, len(rag_overall) + 1)
            
            ax4.plot(question_indices, human_cumsum, '-', label='Human (Cumulative Avg)', linewidth=2)
            ax4.plot(question_indices, rag_cumsum, '-', label='RAG (Cumulative Avg)', linewidth=2)
            ax4.set_xlabel('Question Number')
            ax4.set_ylabel('Cumulative Average Score')
            ax4.set_title('Cumulative Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def analyze(self):
        """Run a complete analysis with all available methods."""
        try:
            print(colored("Generating comprehensive analysis...", "blue"))
            self.print_summary_statistics()
            self.create_radar_chart()
            self._create_mean_comparison_chart()
            self._create_overall_distribution()
            self.create_scatter_plots()
            self.create_violin_plots()
            self._create_boxplots()
            self.create_agreement_analysis()
            self._create_difference_analysis()
            self.create_performance_heatmap()
            self.create_metric_importance_analysis()
            self._create_correlation_analysis()
            self.create_score_evolution_analysis()
            self._create_score_distributions()
            self._perform_statistical_tests()
            print(colored("Comprehensive analysis complete!", "green"))
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
        help="Minimum RAG overall score to include (removes evaluations below threshold).",
    )
    args = parser.parse_args()
    try:
        stats = EvaluationStatistics(results_file=args.file, threshold=args.threshold)
        stats.analyze()
    except FileNotFoundError as e:
        print(colored(f"Error: {e}", "red"))
        print("Run batch evaluation first to generate data.")
    except Exception as e:
        print(colored(f"Unexpected error: {e}", "red"))


if __name__ == "__main__":
    main()

