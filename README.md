# 23CSE301 Machine Learning Capstone: Team 16 (TASKMASTERS)

End-to-end machine learning pipelines for three tracks (regression, classification and clustering), plus a Streamlit app that serves the trained models.

| | |
|---|---|
| Course | 23CSE301 Machine Learning, Academic Year 2026–27 |
| Team | TASKMASTERS (Team 16) |
| Reviews | Review 1: regression + classification Part A. Review 2: classification Part B + clustering |

## Team

| Name | Registration number | Track owned |
|---|---|---|
| Member 1 | [REG_NUMBER] | Regression |
| Member 2 | [REG_NUMBER] | Classification |
| Member 3 | [REG_NUMBER] | Clustering |

## Datasets and problem statements

| Track | File | Rows | Problem |
|---|---|---:|---|
| Regression | `PGA2022.csv` | 1,387 golfer-tournament rows | Predict a golfer's scoring average (`avgScore`) from driving and putting statistics |
| Classification | `AustralianOpen.csv` | 1,905 men's singles matches, 2008–2022 | Predict whether the higher-ranked player wins (`higher_rank_won`), using only information known before the match |
| Clustering | `wta-grand-slam-matches-2018to2022.csv` | 2,413 women's Grand Slam matches | Group matches into playing patterns from serve and return statistics, without labels |

**PGA 2022** (SCORE Network) has driving distance, driving accuracy, putts per round, one-putt %, strokes gained off the tee and strokes gained putting for each golfer at each 2022 PGA Tour event.

**Australian Open** (SCORE Network) has both players' rankings, year turned pro, height, weight and handedness. Rankings are stored as winner/loser columns, which would leak the result if used directly. The notebook rebuilds every feature to be symmetric in the two players: rank gap, favourite's rank, log rank ratio, and experience, height and weight gaps. The target is imbalanced, with favourites winning 74% of matches.

**WTA Grand Slams** has per-match serve statistics for both players (aces, double faults, first-serve and second-serve points won, and break points faced). We remove retirements and walkovers, then turn the raw counts into seven rate features: match length, winner's point share, each player's serve points won, ace rate, double-fault rate and break points per game. Court surface is used only to interpret the clusters after fitting.

## Repository layout

```
├── regression.ipynb          Regression track: 10 algorithms (Review 1)
├── classification.ipynb      Classification track: all 10 algorithms (Part A + Part B)
├── clustering.ipynb          Clustering track: K-Means + agglomerative
├── app.py                    Streamlit app (prediction GUI for all three tracks)
├── .streamlit/config.toml    App theme
├── requirements.txt
├── run_app.bat               Windows shortcut: install requirements and start the app
├── PGA2022.csv
├── AustralianOpen.csv
├── wta-grand-slam-matches-2018to2022.csv
└── 23CSE301_ML_26_27_Capstone_Guidelines.pdf
```

## Environment setup

Python 3.10 or newer.

```bash
git clone https://github.com/Sukanth19/ML-Capstone-Review-1.git
cd ML-Capstone-Review-1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter              # only needed to open the notebooks
```

## How to run

**Notebooks.** Run `jupyter notebook` from the repository root, open a notebook and choose *Run All*. Every notebook runs top to bottom without errors, uses `random_state=42` throughout, and is committed with its outputs.

**App.** Run `streamlit run app.py` (or double-click `run_app.bat` on Windows), then open http://localhost:8501. The first visit to each page trains that track's models, which takes a few seconds; after that they are cached.

| Page | What you can do |
|---|---|
| Classification | Enter two players' ranking, experience, height, weight and handedness, and get the probability that the favourite wins from any of the 10 models. Also shows the comparison table, confusion matrix and ROC curve. |
| Regression | Enter driving and putting statistics and get a predicted scoring average from any of the 10 models. Also shows the comparison table and predicted-vs-actual and residual plots. |
| Clustering | Enter a match's statistics to see which cluster it falls in, with its position on the PCA plot. Also shows the metrics table, elbow curve and dendrograms for each linkage. |

The app uses the hyperparameters selected by GridSearchCV in the notebooks, so its tables match the notebooks.

## Results summary

All results below are on held-out test sets (stratified 80/20 split), copied from the executed notebooks.

### Regression (PGA 2022): ranked by R²

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Support Vector Regressor** (linear, C = 0.1) | **0.268** | **1.284** | **1.065** |
| Linear Regression | 0.263 | 1.288 | 1.070 |
| Ridge Regression | 0.263 | 1.289 | 1.071 |
| Polynomial Regression (degree 2) | 0.262 | 1.290 | 1.065 |
| ElasticNet Regression | 0.259 | 1.292 | 1.075 |
| Lasso Regression | 0.258 | 1.293 | 1.076 |
| KNN Regressor | 0.240 | 1.309 | 1.074 |
| Gradient Boosting Regressor | 0.229 | 1.318 | 1.091 |
| Random Forest Regressor | 0.209 | 1.335 | 1.113 |
| Decision Tree Regressor | 0.077 | 1.442 | 1.167 |

Each row is one golfer at one tournament, so scores vary a lot from week to week. The relationship is mostly linear, which is why the linear models lead. Total strokes gained (`totalSG`) is by far the most important feature.

### Classification (Australian Open): ranked by weighted F1

| Model | Part | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:-:|---:|---:|---:|---:|---:|
| **Gaussian Naive Bayes** | A | 0.737 | 0.723 | 0.737 | **0.729** | 0.696 |
| Random Forest | B | 0.734 | 0.683 | 0.734 | 0.686 | 0.655 |
| MLP Neural Network | B | 0.758 | 0.737 | 0.758 | 0.685 | 0.704 |
| Support Vector Machine | A | 0.753 | 0.718 | 0.753 | 0.678 | 0.640 |
| AdaBoost | B | 0.734 | 0.672 | 0.734 | 0.672 | 0.670 |
| Gradient Boosting | B | 0.718 | 0.658 | 0.718 | 0.670 | 0.653 |
| Bagging | B | 0.711 | 0.655 | 0.711 | 0.670 | 0.635 |
| Logistic Regression | A | 0.753 | 0.743 | 0.753 | 0.663 | 0.705 |
| Decision Tree | A | 0.740 | 0.665 | 0.740 | 0.655 | 0.665 |
| K-Nearest Neighbors | A | 0.695 | 0.635 | 0.695 | 0.654 | 0.600 |
| *Always predict "favourite wins"* | – | 0.745 | 0.555 | 0.745 | 0.636 | 0.500 |

**Final model: Gaussian Naive Bayes.** It has the best weighted F1 both on the test set and in 5-fold CV (0.699 ± 0.015), and it catches the most upsets. Compared with the Review 1 configuration (default parameters, Review 1 features), the two new rank features plus tuning raise weighted F1 by **+0.099** and ROC-AUC by **+0.124**.

### Clustering (WTA Grand Slams): k = 4

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---:|---:|---:|
| **K-Means** | **0.184** | **1.489** | **644.9** |
| Agglomerative (Ward) | 0.133 | 1.726 | 487.7 |

We chose k = 4 with the elbow (kneedle) method and checked it with silhouette scores and cluster sizes. Complete linkage also gives balanced clusters but scores below Ward. Single and average linkage are rejected because they put single outlier matches in their own clusters. The four K-Means clusters are:

| Cluster | Matches | Pattern |
|---|---:|---|
| Serve-dominated | 678 | Highest ace rate, both players hold serve easily |
| Break-heavy | 600 | Many break points, weak serving; over-represented on clay |
| One-sided rout | 434 | Shortest matches; the winner takes 63% of points |
| Long, close battle | 660 | About 190 points; the winner takes only 52% of points |

## Rubric coverage (Review 2)

| Item | Where |
|---|---|
| A1–A3: classification Part B, 10-model table, final model and tuning | `classification.ipynb`, sections 6–10 |
| B1–B3: clustering algorithms, metrics, elbow, dendrogram, PCA, t-SNE | `clustering.ipynb`, sections 4–9 |
| C1: notebooks run top to bottom, modular helpers | All notebooks |
| C2: README | This file |
| C3: commit history, `requirements.txt` | Git history, `requirements.txt` |
| Bonus: GUI that takes inputs and returns predictions | `app.py` |

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit.
