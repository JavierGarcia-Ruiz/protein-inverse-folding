"""Generate EDA and Results Analysis Jupyter notebooks."""

import os
import json


def create_eda_notebook(out_path: str):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# CATH 4.2 Benchmark: Exploratory Data Analysis\n",
                    "**Project**: Autoregressive 3D Structure-to-Sequence Inverse Folding\n",
                    "**Dataset**: CATH 4.2 Non-redundant Domain Structures (<40% Sequence Identity)\n",
                    "**Goal**: Explore structural class topology, Ramachandran dihedral distributions, and residue environments."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from pif.data.constants import CATH_CLASSES, CATH_COLORS, SS_CLASSES, SS_COLORS\n",
                    "\n",
                    "df = pd.read_csv('../data/processed/train.csv')\n",
                    "print(f'Train domain structures: {len(df):,}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Class distribution across CATH architectures\n",
                    "fig, ax = plt.subplots(figsize=(8, 4))\n",
                    "counts = df['cath_class_name'].value_counts()\n",
                    "ax.bar(counts.index, counts.values, color=[CATH_COLORS[i] for i in [1, 2, 3, 4]], edgecolor='black')\n",
                    "ax.set_title('CATH Superfamily Structural Class Distribution')\n",
                    "ax.set_ylabel('Number of Domains')\n",
                    "plt.xticks(rotation=15)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Secondary structure fractions across folds\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
                    "sns.boxplot(data=df, x='cath_class_name', y='helix_fraction', ax=axes[0], palette='Blues')\n",
                    "axes[0].set_title('Alpha-Helix Fraction')\n",
                    "axes[0].tick_params(axis='x', rotation=20)\n",
                    "\n",
                    "sns.boxplot(data=df, x='cath_class_name', y='sheet_fraction', ax=axes[1], palette='Greens')\n",
                    "axes[1].set_title('Beta-Sheet Fraction')\n",
                    "axes[1].tick_params(axis='x', rotation=20)\n",
                    "\n",
                    "sns.boxplot(data=df, x='cath_class_name', y='loop_fraction', ax=axes[2], palette='Reds')\n",
                    "axes[2].set_title('Loop / Coil Fraction')\n",
                    "axes[2].tick_params(axis='x', rotation=20)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def create_results_notebook(out_path: str):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Inverse Protein Folding: Benchmark Results & Generative Sequence Design\n",
                    "**Project**: 3D Backbone Inverse Folding via Message Passing Neural Networks\n",
                    "**Comparison**: FrequencyBaseline vs SequenceTransformer vs ProteinMPNN"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "metrics_df = pd.read_csv('../reports/results/metrics_benchmarks.csv')\n",
                    "metrics_df"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Sequence Recovery vs Perplexity\n",
                    "fig, ax1 = plt.subplots(figsize=(8, 4.5))\n",
                    "ax2 = ax1.twinx()\n",
                    "x = range(len(metrics_df))\n",
                    "\n",
                    "ax1.bar([i - 0.2 for i in x], metrics_df['sequence_recovery'], width=0.4, label='Recovery (%)', color='#009E73')\n",
                    "ax2.bar([i + 0.2 for i in x], metrics_df['perplexity'], width=0.4, label='Perplexity (PPL)', color='#D55E00')\n",
                    "\n",
                    "ax1.set_xticks(x)\n",
                    "ax1.set_xticklabels(metrics_df['model'])\n",
                    "ax1.set_ylabel('Sequence Recovery (%)', color='#009E73')\n",
                    "ax2.set_ylabel('Perplexity (PPL)', color='#D55E00')\n",
                    "plt.title('Inverse Folding Model Comparison on Held-Out Test Folds')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def main():
    nb_dir = "notebooks"
    os.makedirs(nb_dir, exist_ok=True)
    create_eda_notebook(os.path.join(nb_dir, "01_eda.ipynb"))
    create_results_notebook(os.path.join(nb_dir, "02_results_analysis.ipynb"))
    print("Generated 01_eda.ipynb and 02_results_analysis.ipynb successfully!")


if __name__ == "__main__":
    main()

