"""Adoption-outcome model trained on REAL Austin Animal Center records (data/austin_outcomes_labeled.csv).
Target: outcome == Adoption (vs Transfer, Return_to_owner, Euthanasia, Died). Time-based split (last 20% by date).
Explanations: occlusion attribution (feature replaced by its typical value), a SHAP-style local explanation."""
import os, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
D = os.path.join(os.path.dirname(__file__), "..", "..", "data"); PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
NUM, CAT = ["age_months", "altered", "has_name", "is_mix"], ["species", "sex", "breed", "color"]
LABELS = {"age_months": "Age", "altered": "Spayed/neutered", "has_name": "Has a name", "is_mix": "Mixed breed", "species": "Species",
          "sex": "Sex", "breed": "Breed", "color": "Coat color"}
def parse_age(s):
    if pd.isna(s): return np.nan
    try: n, u = str(s).split()[:2]; n = int(n)
    except ValueError: return np.nan
    return n * (12 if u.startswith("year") else 1 if u.startswith("month") else .25 if u.startswith("week") else 1 / 30 if u.startswith("day") else np.nan)
def prep(df):
    s = df["SexuponOutcome"].fillna("Unknown")
    return pd.DataFrame({"age_months": df["AgeuponOutcome"].map(parse_age),
        "altered": s.map(lambda x: 1.0 if x.startswith(("Neutered", "Spayed")) else 0.0 if x.startswith("Intact") else np.nan),
        "has_name": df["Name"].notna().astype(float), "is_mix": df["Breed"].str.contains("Mix|/", regex=True).astype(float),
        "species": df["AnimalType"], "sex": s.map(lambda x: "Male" if "Male" in x else "Female" if "Female" in x else "Unknown"),
        "breed": df["Breed"].str.split("/").str[0].str.replace(" Mix", "", regex=False).str.strip(),
        "color": df["Color"].str.split("/").str[0].str.split(" ").str[0]})
def labeled():
    d = pd.read_csv(os.path.join(D, "austin_outcomes_labeled.csv"), parse_dates=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return d
def pipe(est):
    pre = ColumnTransformer([("n", SimpleImputer(strategy="median"), NUM), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), CAT)])
    return make_pipeline(pre, est)
def train():
    d = labeled(); X, y = prep(d), (d.OutcomeType == "Adoption").astype(int); cut = int(len(d) * .8)
    cands = {"LogisticRegression": LogisticRegression(max_iter=1000), "RandomForest": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=1),
             "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=.06, random_state=1)}
    res, fitted = {}, {}
    for n, e in cands.items():
        m = pipe(e).fit(X[:cut], y[:cut]); pr = m.predict_proba(X[cut:])[:, 1]; pd_ = (pr >= .5).astype(int); yt = y[cut:]
        res[n] = dict(accuracy=accuracy_score(yt, pd_), precision=precision_score(yt, pd_), recall=recall_score(yt, pd_), f1=f1_score(yt, pd_),
                      roc_auc=roc_auc_score(yt, pr), pr_auc=average_precision_score(yt, pr)); fitted[n] = m
    best = max(res, key=lambda k: res[k]["roc_auc"])
    base = {c: float(X[c].median()) for c in NUM} | {c: X[c].mode()[0] for c in CAT}
    joblib.dump({"model": fitted[best], "name": best, "base": base, "metrics": res[best], "all": res, "n_train": cut, "n_test": len(d) - cut,
                 "base_rate": float(y.mean())}, PATH)
    return best, res
_b = None
def bundle():
    global _b
    if _b is None:
        if not os.path.exists(PATH): train()
        _b = joblib.load(PATH)
    return _b
def frame(pets):
    return pd.DataFrame([{"age_months": p["age_months"], "altered": (np.nan if p.get("altered") is None else float(p["altered"])), "has_name": float(bool(p.get("has_name"))),
        "is_mix": float(bool(p.get("is_mix"))), "species": p["species"], "sex": p["sex"], "breed": p["breed"], "color": p["color"]} for p in pets])
def predict_many(pets):
    return list(bundle()["model"].predict_proba(frame(pets))[:, 1]) if pets else []
def risk(p): return "LOW" if p >= .5 else "MEDIUM" if p >= .3 else "HIGH"
def predict(p, explain=True):
    b = bundle(); X = frame([p]); prob = float(b["model"].predict_proba(X)[0, 1])
    out = {"probability": round(prob, 3), "risk": risk(prob), "baseline": round(b["base_rate"], 3), "model": b["name"]}
    if explain:
        rows = []
        for c in NUM + CAT:
            X2 = X.copy(); X2[c] = b["base"][c]; rows.append((c, X2))
        probs = b["model"].predict_proba(pd.concat([r[1] for r in rows]))[:, 1]
        out["factors"] = sorted([{"feature": LABELS[c], "impact": round(prob - float(q), 3)} for (c, _), q in zip(rows, probs)], key=lambda f: -abs(f["impact"]))[:6]
    return out
def stats():
    """Real aggregates from the dataset."""
    d = labeled(); X = prep(d); d["adopted"] = d.OutcomeType == "Adoption"
    grp = pd.cut(X.age_months, [-1, 11.9, 35.9, 83.9, 999], labels=["<1 yr", "1-2 yrs", "3-6 yrs", "7+ yrs"])
    d["month"] = d.DateTime.dt.to_period("M"); m = d[d.adopted].groupby(["month", "AnimalType"]).size().unstack(fill_value=0)
    m = m.iloc[:-1]  # last month is partial
    fc = {}
    for sp in m.columns:
        y = m[sp].values.astype(float); t = np.arange(len(y)).reshape(-1, 1); lr = LinearRegression().fit(t, y); sd = float(np.std(y - lr.predict(t)))
        fut = lr.predict(np.arange(len(y), len(y) + 3).reshape(-1, 1))
        fc[sp] = {"actual": y.tolist(), "forecast": [round(max(v, 0), 1) for v in fut], "ci": round(1.96 * sd, 1)}
    return {"records": len(d), "date_range": [str(d.DateTime.min().date()), str(d.DateTime.max().date())], "months": [str(x) for x in m.index],
            "outcomes": d.OutcomeType.value_counts().to_dict(), "adoption_rate_by_age": (d.groupby(grp, observed=True).adopted.mean().round(3)).to_dict(),
            "adoption_rate_by_species": d.groupby("AnimalType").adopted.mean().round(3).to_dict(),
            "adoption_rate_by_altered": {("Spayed/neutered" if k == 1 else "Intact" if k == 0 else "Unknown"): round(v, 3) for k, v in d.groupby(X.altered.fillna(-1)).adopted.mean().items()},
            "forecast": fc}
if __name__ == "__main__":
    b, r = train(); print("best:", b)
    for k, v in r.items(): print(k, {m: round(x, 3) for m, x in v.items()})
