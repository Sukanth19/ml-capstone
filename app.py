"""
Streamlit front end for the 23CSE301 ML capstone (Team 16).

Each track page trains the models from the notebooks once (cached), shows the
test-set comparison, and has a Predict tab that turns user inputs into a
prediction from the trained model.

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42

st.set_page_config(page_title="ML Capstone · Team 16", layout="wide")

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})
PALETTE = plt.get_cmap("tab10").colors


def preprocessor(features, scale=True, poly_degree=None):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if poly_degree:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("numeric", Pipeline(steps), features)])


def pipe(features, model, **prep_kwargs):
    return Pipeline([("preprocess", preprocessor(features, **prep_kwargs)), ("model", model)])


# ---------------------------------------------------------------- classification

CLS_FEATURES = ["rank_gap_abs", "experience_gap_abs", "height_gap_abs", "weight_gap_abs",
                "same_handedness_bin", "fav_rank", "log_rank_ratio"]
CLS_FINAL_MODEL = "Gaussian Naive Bayes"


def match_features(rank_a, rank_b, pro_a, pro_b, height_a, height_b, weight_a, weight_b, same_hand):
    # Same symmetric features as classification.ipynb, section 3.
    favourite, underdog = min(rank_a, rank_b), max(rank_a, rank_b)
    return {
        "rank_gap_abs": underdog - favourite,
        "experience_gap_abs": abs(pro_a - pro_b),
        "height_gap_abs": abs(height_a - height_b),
        "weight_gap_abs": abs(weight_a - weight_b),
        "same_handedness_bin": int(same_hand),
        "fav_rank": favourite,
        "log_rank_ratio": np.log(underdog / favourite),
    }


@st.cache_data
def load_tennis():
    df = pd.read_csv("AustralianOpen.csv", encoding="latin1").dropna(subset=["higher_rank_won"])
    rows = [
        match_features(r.WRank, r.LRank, r.pl1_year_pro, r.pl2_year_pro, r.pl1_height,
                       r.pl2_height, r.pl1_weight, r.pl2_weight, r.same_handedness == "Yes")
        for r in df.itertuples()
    ]
    X = pd.DataFrame(rows, index=df.index)
    # Missing player-2 profiles must stay missing so the pipeline imputes them.
    missing_profile = df["pl2_year_pro"].isna()
    X.loc[missing_profile, ["experience_gap_abs", "height_gap_abs", "weight_gap_abs",
                            "same_handedness_bin"]] = np.nan
    y = df["higher_rank_won"].eq("Yes").astype(int)
    return df, X, y


def classifiers():
    # Hyperparameters are the GridSearchCV results from classification.ipynb.
    f = CLS_FEATURES
    return {
        "Logistic Regression": pipe(f, LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)),
        "K-Nearest Neighbors": pipe(f, KNeighborsClassifier(n_neighbors=5, metric="manhattan")),
        "Gaussian Naive Bayes": pipe(f, GaussianNB(var_smoothing=0.01)),
        "Decision Tree": pipe(f, DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=20,
                                                        random_state=RANDOM_STATE)),
        "Support Vector Machine": pipe(f, SVC(C=10, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        "Random Forest": pipe(f, RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        "AdaBoost": pipe(f, AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(f, GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                                                random_state=RANDOM_STATE)),
        "Bagging": pipe(f, BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                             n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        "MLP Neural Network": pipe(f, MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", alpha=0.01,
                                                    max_iter=1000, early_stopping=True,
                                                    random_state=RANDOM_STATE)),
    }


@st.cache_resource(show_spinner="Training the 10 classifiers…")
def train_classifiers():
    _, X, y = load_tennis()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    models, rows, outputs = {}, [], {}
    for name, model in classifiers().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        models[name] = model
        outputs[name] = (pred, prob)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_test, pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })
    table = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False).set_index("Model")
    return models, table, outputs, y_test


def classification_page():
    st.header("Classification: will the favourite win?")
    st.caption("Australian Open men's singles, 2008–2022. Target: did the higher-ranked player win? "
               "All features are symmetric in the two players, so the result of the match never leaks in.")

    df, X, _ = load_tennis()
    models, table, outputs, y_test = train_classifiers()

    predict_tab, compare_tab, data_tab = st.tabs(["Predict", "Model comparison", "Data"])

    with predict_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Player A")
            rank_a = st.number_input("ATP ranking", 1, 2000, 8, key="rank_a")
            pro_a = st.number_input("Year turned pro", 1990, 2025, 2012, key="pro_a")
            height_a = st.number_input("Height (cm)", 160, 215, 188, key="height_a")
            weight_a = st.number_input("Weight (kg)", 55, 120, 82, key="weight_a")
            hand_a = st.radio("Plays", ["Right", "Left"], horizontal=True, key="hand_a")
        with right:
            st.subheader("Player B")
            rank_b = st.number_input("ATP ranking", 1, 2000, 64, key="rank_b")
            pro_b = st.number_input("Year turned pro", 1990, 2025, 2016, key="pro_b")
            height_b = st.number_input("Height (cm)", 160, 215, 185, key="height_b")
            weight_b = st.number_input("Weight (kg)", 55, 120, 80, key="weight_b")
            hand_b = st.radio("Plays", ["Right", "Left"], horizontal=True, key="hand_b")

        names = list(models)
        model_name = st.selectbox("Model", names, index=names.index(CLS_FINAL_MODEL),
                                  help="Gaussian Naive Bayes is the final model chosen in the notebook.")

        features = match_features(rank_a, rank_b, pro_a, pro_b, height_a, height_b,
                                  weight_a, weight_b, hand_a == hand_b)
        p_fav = models[model_name].predict_proba(pd.DataFrame([features])[CLS_FEATURES])[0, 1]

        if rank_a == rank_b:
            st.info("Both players have the same ranking, so there is no favourite. "
                    "The model treats Player A as the favourite.")
        favourite, fav_rank = ("Player A", rank_a) if rank_a <= rank_b else ("Player B", rank_b)
        underdog = "Player B" if favourite == "Player A" else "Player A"
        winner = favourite if p_fav >= 0.5 else underdog

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Favourite", f"{favourite} (#{fav_rank})")
        c2.metric("P(favourite wins)", "> 99.9%" if p_fav > 0.999 else f"{p_fav:.1%}")
        c3.metric("Predicted winner", winner)
        st.caption(f"Base rate in the data: the favourite wins {y_test.mean():.0%} of matches.")

        with st.expander("Features passed to the model"):
            st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "value"}), width="stretch")

    with compare_tab:
        st.subheader("Test-set results (380 matches, stratified 80/20 split)")
        st.dataframe(table.style.format("{:.4f}").highlight_max(color="#d9ead3"), width="stretch")

        chosen = st.selectbox("Inspect a model", table.index, key="cls_inspect")
        pred, prob = outputs[chosen]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(4, 3.4))
            cm = confusion_matrix(y_test, pred)
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["Upset", "Fav. won"])
            ax.set_yticks([0, 1], ["Upset", "Fav. won"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion matrix: {chosen}")
            ax.grid(False)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 3.4))
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, color=PALETTE[0], label=f"AUC = {table.loc[chosen, 'ROC-AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"ROC curve: {chosen}")
            ax.legend(loc="lower right")
            st.pyplot(fig)

    with data_tab:
        st.write(f"{len(df):,} matches with a known result.")
        st.dataframe(df.head(200), width="stretch")


# ---------------------------------------------------------------- regression

REG_INPUTS = {
    "avgDriveDist": ("Average driving distance (yards)", 1.0),
    "drivePct": ("Driving accuracy (% fairways hit)", 0.5),
    "driveSG": ("Strokes gained: off the tee", 0.05),
    "avgPuttsPerRound": ("Putts per round", 0.25),
    "onePuttPct": ("One-putt percentage", 0.5),
    "puttsSG": ("Strokes gained: putting", 0.05),
}
REG_FEATURES = list(REG_INPUTS) + ["totalSG"]


@st.cache_data
def load_golf():
    df = pd.read_csv("PGA2022.csv", encoding="latin1").dropna(subset=["avgScore"])
    df["totalSG"] = df["driveSG"] + df["puttsSG"]
    return df


def regressors():
    # Hyperparameters are the GridSearchCV results from regression.ipynb.
    f = REG_FEATURES
    return {
        "Linear Regression": pipe(f, LinearRegression()),
        "Ridge Regression": pipe(f, Ridge(alpha=10)),
        "Lasso Regression": pipe(f, Lasso(alpha=0.01, max_iter=20000)),
        "ElasticNet Regression": pipe(f, ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=20000)),
        "Polynomial Regression": pipe(f, LinearRegression(), poly_degree=2),
        "Decision Tree": pipe(f, DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE), scale=False),
        "Random Forest": pipe(f, RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE,
                                                       n_jobs=-1), scale=False),
        "Gradient Boosting": pipe(f, GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2,
                                                               random_state=RANDOM_STATE), scale=False),
        "Support Vector Regressor": pipe(f, SVR(C=0.1, kernel="linear")),
        "KNN Regressor": pipe(f, KNeighborsRegressor(n_neighbors=21)),
    }


@st.cache_resource(show_spinner="Training the 10 regressors…")
def train_regressors():
    df = load_golf()
    X, y = df[REG_FEATURES], df["avgScore"]
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=bins)
    models, rows, preds = {}, [], {}
    for name, model in regressors().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name], preds[name] = model, pred
        rows.append({"Model": name, "R²": r2_score(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "MAE": mean_absolute_error(y_test, pred)})
    table = pd.DataFrame(rows).sort_values("R²", ascending=False).set_index("Model")
    return models, table, preds, y_test


def regression_page():
    st.header("Regression: predict a golfer's scoring average")
    st.caption("PGA Tour 2022, one row per golfer per tournament. Target: average strokes per round (avgScore).")

    df = load_golf()
    models, table, preds, y_test = train_regressors()

    predict_tab, compare_tab, data_tab = st.tabs(["Predict", "Model comparison", "Data"])

    with predict_tab:
        values = {}
        cols = st.columns(3)
        for i, (feature, (label, step)) in enumerate(REG_INPUTS.items()):
            col = df[feature]
            values[feature] = cols[i % 3].number_input(
                label, float(col.min()), float(col.max()), float(round(col.median(), 2)), step=step,
                key=f"reg_{feature}")
        values["totalSG"] = values["driveSG"] + values["puttsSG"]

        names = list(table.index)
        model_name = st.selectbox("Model", names, index=0, help="Sorted by test-set R²; the first one is the best.")
        score = models[model_name].predict(pd.DataFrame([values])[REG_FEATURES])[0]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted scoring average", f"{score:.2f}")
        c2.metric("Field median", f"{df['avgScore'].median():.2f}",
                  delta=f"{score - df['avgScore'].median():+.2f} strokes", delta_color="inverse")
        c3.metric("Strokes gained, total", f"{values['totalSG']:+.2f}")
        st.caption(f"Typical error of this model on unseen golfers (MAE): ±{table.loc[model_name, 'MAE']:.2f} strokes.")

    with compare_tab:
        st.subheader("Test-set results (stratified 80/20 split)")
        st.dataframe(table.style.format("{:.4f}"), width="stretch")

        chosen = st.selectbox("Inspect a model", table.index, key="reg_inspect")
        pred = preds[chosen]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(4, 3.4))
            ax.scatter(y_test, pred, s=10, alpha=0.6, color=PALETTE[0])
            lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
            ax.plot(lims, lims, "--", color="grey", label="Perfect prediction")
            ax.set_xlabel("Actual avgScore")
            ax.set_ylabel("Predicted avgScore")
            ax.set_title(f"Predicted vs actual: {chosen}")
            ax.legend()
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 3.4))
            ax.scatter(pred, y_test - pred, s=10, alpha=0.6, color=PALETTE[1])
            ax.axhline(0, color="grey", linestyle="--")
            ax.set_xlabel("Predicted avgScore")
            ax.set_ylabel("Residual (actual − predicted)")
            ax.set_title(f"Residuals: {chosen}")
            st.pyplot(fig)

    with data_tab:
        st.write(f"{len(df):,} golfer-tournament rows.")
        st.dataframe(df.head(200), width="stretch")


# ---------------------------------------------------------------- clustering

CLU_FEATURES = ["total_points", "winner_point_share", "winner_serve_won_pct", "loser_serve_won_pct",
                "ace_rate", "df_rate", "bp_per_game"]
CLU_K = 4


@st.cache_data
def load_wta():
    # Same cleaning and features as clustering.ipynb, section 2.
    df = pd.read_csv("wta-grand-slam-matches-2018to2022.csv")
    m = df[~df["score"].str.contains("RET|W/O|DEF", na=False)]
    m = m.dropna(subset=["w_svpt", "l_svpt", "w_SvGms", "l_SvGms"])
    m = m[(m["w_svpt"] > 0) & (m["l_svpt"] > 0)].copy()

    total = m["w_svpt"] + m["l_svpt"]
    w_won = m["w_1stWon"] + m["w_2ndWon"]
    l_won = m["l_1stWon"] + m["l_2ndWon"]
    feats = pd.DataFrame({
        "total_points": total,
        "winner_point_share": (w_won + (m["l_svpt"] - l_won)) / total,
        "winner_serve_won_pct": w_won / m["w_svpt"],
        "loser_serve_won_pct": l_won / m["l_svpt"],
        "ace_rate": (m["w_ace"] + m["l_ace"]) / total,
        "df_rate": (m["w_df"] + m["l_df"]) / total,
        "bp_per_game": (m["w_bpFaced"] + m["l_bpFaced"]) / (m["w_SvGms"] + m["l_SvGms"]),
    }, index=m.index).replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median())
    return m, feats


@st.cache_resource(show_spinner="Fitting the clustering models…")
def fit_clustering():
    matches, feats = load_wta()
    lower, upper = feats.quantile(0.01), feats.quantile(0.99)
    scaler = StandardScaler().fit(feats.clip(lower, upper, axis=1))
    X = scaler.transform(feats.clip(lower, upper, axis=1))

    kmeans = KMeans(n_clusters=CLU_K, n_init=10, random_state=RANDOM_STATE).fit(X)
    ward = AgglomerativeClustering(n_clusters=CLU_K, linkage="ward").fit_predict(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)

    # Name clusters from their profile rather than their (arbitrary) label number.
    profile = feats.assign(cluster=kmeans.labels_).groupby("cluster").mean()
    shortest, longest = profile["total_points"].idxmin(), profile["total_points"].idxmax()
    rest = profile.drop([shortest, longest])
    serve = rest["ace_rate"].idxmax()
    names = {shortest: "One-sided rout", longest: "Long, close battle", serve: "Serve-dominated",
             rest.drop(serve).index[0]: "Break-heavy"}

    def scores(labels):
        return {"Silhouette": silhouette_score(X, labels), "Davies-Bouldin": davies_bouldin_score(X, labels),
                "Calinski-Harabasz": calinski_harabasz_score(X, labels)}

    metrics = pd.DataFrame({"K-Means": scores(kmeans.labels_), "Agglomerative (Ward)": scores(ward)}).T
    return dict(X=X, lower=lower, upper=upper, scaler=scaler, kmeans=kmeans, ward=ward, pca=pca,
                names=names, profile=profile, metrics=metrics, surface=matches["surface"].values)


@st.cache_data(show_spinner="Running K-Means for k = 2…10…")
def elbow_curve():
    X = fit_clustering()["X"]
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def clustering_page():
    st.header("Clustering: what kinds of matches are there?")
    st.caption("WTA Grand Slam matches, 2018–2022. Matches are grouped by how they were played "
               "(length, one-sidedness, serve and break-point rates). Surface is not used for fitting.")

    _, feats = load_wta()
    c = fit_clustering()

    predict_tab, compare_tab, explore_tab = st.tabs(["Assign a match", "Model comparison", "Elbow & dendrogram"])

    with predict_tab:
        st.write("Enter a match's statistics to see which K-Means cluster it belongs to.")
        cols = st.columns(4)
        total_points = cols[0].number_input("Total points played", 40, 320, 140, step=5)
        winner_share = cols[1].number_input("Winner's share of all points (%)", 40.0, 85.0, 56.0, step=0.5)
        w_serve = cols[2].number_input("Winner's serve points won (%)", 35.0, 95.0, 64.0, step=0.5)
        l_serve = cols[3].number_input("Loser's serve points won (%)", 20.0, 75.0, 51.0, step=0.5)
        cols = st.columns(4)
        aces = cols[0].number_input("Aces per 100 points", 0.0, 20.0, 4.0, step=0.5)
        dfs = cols[1].number_input("Double faults per 100 points", 0.0, 16.0, 5.0, step=0.5)
        bps = cols[2].number_input("Break points faced per service game", 0.0, 3.0, 0.75, step=0.05)

        new = pd.DataFrame([{
            "total_points": total_points, "winner_point_share": winner_share / 100,
            "winner_serve_won_pct": w_serve / 100, "loser_serve_won_pct": l_serve / 100,
            "ace_rate": aces / 100, "df_rate": dfs / 100, "bp_per_game": bps,
        }])[CLU_FEATURES]
        z = c["scaler"].transform(new.clip(c["lower"], c["upper"], axis=1))
        cluster = int(c["kmeans"].predict(z)[0])
        distances = np.linalg.norm(c["kmeans"].cluster_centers_ - z, axis=1)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Cluster", f"{cluster}: {c['names'][cluster]}")
        m2.metric("Distance to its centre", f"{distances[cluster]:.2f}")
        runner_up = int(np.argsort(distances)[1])
        m3.metric("Next closest", f"{runner_up}: {c['names'][runner_up]}")

        left, right = st.columns([3, 2])
        with left:
            pca, labels = c["pca"], c["kmeans"].labels_
            coords, point = pca.transform(c["X"]), pca.transform(z)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for k in range(CLU_K):
                mask = labels == k
                ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.35, color=PALETTE[k],
                           label=f"{k}: {c['names'][k]}")
            ax.scatter(point[:, 0], point[:, 1], marker="*", s=320, color="black", label="Your match")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
            ax.set_title("Your match among all matches (PCA)")
            ax.legend(fontsize=7, loc="best")
            st.pyplot(fig)
        with right:
            st.write("Average match in each cluster")
            profile = c["profile"].copy()
            profile.index = [f"{k}: {c['names'][k]}" for k in profile.index]
            st.dataframe(profile.T.style.format("{:.3f}"), width="stretch")

    with compare_tab:
        st.subheader(f"Internal metrics, k = {CLU_K}")
        st.dataframe(c["metrics"].style.format("{:.4f}"), width="stretch")
        st.caption("Silhouette: higher is better. Davies-Bouldin: lower is better. "
                   "Calinski-Harabasz: higher is better.")

        algo = st.radio("Colour points by", ["K-Means", "Agglomerative (Ward)"], horizontal=True)
        labels = c["kmeans"].labels_ if algo == "K-Means" else c["ward"]
        coords = c["pca"].transform(c["X"])
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for k in np.unique(labels):
            mask = labels == k
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5, color=PALETTE[k],
                       label=f"Cluster {k} (n={mask.sum()})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{algo}: PCA projection")
        ax.legend(fontsize=7)
        st.pyplot(fig)

        surface = pd.crosstab(pd.Series(c["kmeans"].labels_, name="Cluster").map(
            lambda k: f"{k}: {c['names'][k]}"), pd.Series(c["surface"], name="Surface"), normalize="columns")
        st.write("Share of each surface's matches per K-Means cluster (surface was only used after fitting):")
        st.dataframe(surface.style.format("{:.0%}"), width="stretch")

    with explore_tab:
        elbow = elbow_curve()
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.plot(elbow.index, elbow["Inertia"], marker="o", color=PALETTE[0])
            ax.axvline(CLU_K, color="grey", linestyle="--", label=f"Chosen k = {CLU_K}")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow curve")
            ax.legend()
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.plot(elbow.index, elbow["Silhouette"], marker="o", color=PALETTE[2])
            ax.axvline(CLU_K, color="grey", linestyle="--", label=f"Chosen k = {CLU_K}")
            ax.set_xlabel("k")
            ax.set_ylabel("Silhouette score")
            ax.set_title("Silhouette vs k")
            ax.legend()
            st.pyplot(fig)

        method = st.selectbox("Dendrogram linkage", ["ward", "complete", "average", "single"])
        Z = linkage(c["X"], method=method)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90, color_threshold=Z[-(CLU_K - 1), 2])
        ax.set_xlabel("Cluster (number of matches)")
        ax.set_ylabel("Merge distance")
        ax.set_title(f"Dendrogram, {method} linkage (last 30 merges)")
        ax.grid(False)
        st.pyplot(fig)


# ---------------------------------------------------------------- about

def about_page():
    st.header("About")
    st.markdown(
        """
23CSE301 Machine Learning capstone, Team 16 (TASKMASTERS).

| Track | Dataset | Task | Final model |
|---|---|---|---|
| Regression | PGA Tour 2022 | Predict scoring average | Support Vector Regressor (linear) |
| Classification | Australian Open 2008–2022 | Did the higher-ranked player win? | Gaussian Naive Bayes |
| Clustering | WTA Grand Slams 2018–2022 | Group matches by playing pattern | K-Means, k = 4 |

The full analysis, including EDA, tuning and every required plot, is in the three notebooks.
This app retrains the same models with the hyperparameters the notebooks selected,
so the numbers here match the notebooks.
"""
    )


PAGES = {
    "Classification": classification_page,
    "Regression": regression_page,
    "Clustering": clustering_page,
    "About": about_page,
}

st.sidebar.title("ML Capstone")
st.sidebar.caption("23CSE301 · Team 16")
page = st.sidebar.radio("Track", list(PAGES), label_visibility="collapsed")
PAGES[page]()
