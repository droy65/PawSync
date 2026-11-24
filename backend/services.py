"""Business logic: serialization, recommendations, NL intent extraction, priority scoring (real fields only)."""
import re, datetime as dt, numpy as np
from bson import ObjectId
from ml import model as mlm
def ser(d):
    if isinstance(d, list): return [ser(x) for x in d]
    if isinstance(d, dict): return {k: ser(v) for k, v in d.items() if k != "password_hash"}
    if isinstance(d, ObjectId): return str(d)
    if isinstance(d, dt.datetime): return d.isoformat()
    return d
def now(): return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
def age_group(m): return "Young" if m < 24 else "Adult" if m < 84 else "Senior"
def tokens(p):
    return [f"sp:{p['species']}", f"ag:{age_group(p['age_months'])}", f"sx:{p['sex']}", f"br:{p['breed'].lower()}"]
def user_tokens(u, favs):
    pr = u.get("preferences", {}); t = []
    for k, pre in (("species", "sp"), ("age_group", "ag"), ("sex", "sx")):
        if pr.get(k): t.append(f"{pre}:{pr[k]}")
    if pr.get("breed"): t.append(f"br:{pr['breed'].lower()}")
    for p in favs: t += tokens(p)
    return t
def recommend(db, user, k=6):
    """Hybrid: cosine similarity of preference/favorite profile vs pet + model adoption likelihood (hard-to-place boost)."""
    favs = [f for f in (db.pets.find_one({"_id": x["pet_id"]}) for x in db.favorites.find({"user_id": user["_id"]})) if f]
    viewed = [db.pets.find_one({"_id": x["pet_id"]}) for x in db.interactions.find({"user_id": user["_id"], "event": "pet_view", "pet_id": {"$exists": True}}).limit(30)]
    ut = user_tokens(user, favs + [v for v in viewed if v]); pets = list(db.pets.find({"status": "Available"}))
    vocab = sorted({t for p in pets for t in tokens(p)} | set(ut)); ix = {t: i for i, t in enumerate(vocab)}
    def v(ts):
        a = np.zeros(len(vocab))
        for t in ts: a[ix[t]] += 1
        return a
    uv = v(ut); probs = mlm.predict_many(pets); pr = user.get("preferences", {}); out = []
    for p, pb in zip(pets, probs):
        pv = v(tokens(p)); cos = float(uv @ pv / (np.linalg.norm(uv) * np.linalg.norm(pv) + 1e-9))
        why = []
        if pr.get("species") == p["species"]: why.append("Matches your preferred species")
        if pr.get("age_group") == age_group(p["age_months"]): why.append("Matches your preferred age group")
        if pr.get("sex") == p["sex"]: why.append("Matches your preferred sex")
        if pr.get("breed") and pr["breed"].lower() in p["breed"].lower(): why.append("Matches your preferred breed")
        if any(f["species"] == p["species"] and f["breed"] == p["breed"] for f in favs): why.append("Same breed as a pet you favorited")
        if pb < .35: why.append("Has a lower predicted adoption likelihood, so a home matters more")
        out.append((0.85 * cos + 0.15 * (1 - pb), p, why, pb))
    out.sort(key=lambda x: -x[0])
    return [dict(ser(p), match=round(min(s, 1) * 100), why=w, prediction={"probability": round(float(pb), 3), "risk": mlm.risk(pb)}) for s, p, w, pb in out[:k]]
def extract_intent(text, breeds=()):
    """Rule-based intent extractor. Swap in an LLM here (LLM_API_KEY); output schema stays the same."""
    t = text.lower(); q = {}
    if re.search(r"\b(dog|puppy|puppies)\b", t): q["species"] = "Dog"
    if re.search(r"\b(cat|kitten)s?\b", t): q["species"] = "Cat"
    if re.search(r"puppy|puppies|kitten|baby|young", t): q["age_group"] = "Young"
    elif re.search(r"senior|old|elderly", t): q["age_group"] = "Senior"
    elif "adult" in t: q["age_group"] = "Adult"
    m = re.search(r"under (\d+) years?", t)
    if m: q["max_age_months"] = int(m.group(1)) * 12
    if re.search(r"\bfemale\b|\bgirl\b", t): q["sex"] = "Female"
    elif re.search(r"\bmale\b|\bboy\b", t): q["sex"] = "Male"
    if re.search(r"neutered|spayed|fixed", t): q["altered"] = "1"
    for b in sorted(breeds, key=len, reverse=True):
        if b.lower() in t: q["breed"] = b; break
    return q
AGE = {"Young": {"$lt": 24}, "Adult": {"$gte": 24, "$lt": 84}, "Senior": {"$gte": 84}}
def build_query(f):
    q = {"status": "Available"}
    for k in ("species", "sex"):
        if f.get(k): q[k] = f[k]
    if f.get("breed"): q["breed"] = {"$regex": re.escape(f["breed"]), "$options": "i"}
    if f.get("q"): q["name"] = {"$regex": re.escape(f["q"]), "$options": "i"}
    if f.get("age_group") in AGE: q["age_months"] = dict(AGE[f["age_group"]])
    if f.get("max_age_months"): q.setdefault("age_months", {})["$lte"] = int(f["max_age_months"])
    if f.get("altered") in ("1", 1, True): q["altered"] = True
    return q
def priority(p, prob):
    s = (1 - prob) * 70 + min(p["age_months"] / 120, 1) * 15 + (0 if p.get("altered") else 10) + (0 if p.get("has_name") else 5)
    return round(s), "Critical" if s >= 55 else "High" if s >= 45 else "Medium" if s >= 35 else "Low"
