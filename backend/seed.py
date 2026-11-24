"""Loads REAL Austin Animal Center animals (data/austin_animals_unlabeled.csv, outcome not yet known) into MongoDB.
Usage: python backend/seed.py   - no fake users/applications are created; only 3 demo accounts."""
import os
import pandas as pd
from werkzeug.security import generate_password_hash
from services import now
from ml import model as mlm
N_PETS = int(os.getenv("SEED_PETS", 400))
def seed(db, force=False):
    if db.pets.count_documents({}) and not force: return
    if force:
        for c in db.list_collection_names(): db[c].delete_many({})
    df = pd.read_csv(os.path.join(mlm.D, "austin_animals_unlabeled.csv")).sample(N_PETS, random_state=42).reset_index(drop=True)
    X = mlm.prep(df)
    sid = db.shelters.insert_one({"name": "Austin Animal Center", "city": "Austin", "state": "TX", "verified": True,
        "description": "Source of the real records used in this project (City of Austin open data).", "created_at": now()}).inserted_id
    docs = []
    for (_, r), (_, x) in zip(df.iterrows(), X.iterrows()):
        if pd.isna(x.age_months): continue
        nm = r["Name"] if pd.notna(r["Name"]) else None
        docs.append({"name": (nm or f"Unnamed {r['AnimalType']}").lstrip("*"), "has_name": nm is not None, "species": r["AnimalType"], "breed": x.breed, "is_mix": bool(x.is_mix),
            "age_months": round(float(x.age_months), 1), "sex": x.sex, "altered": None if pd.isna(x.altered) else bool(x.altered), "color": r["Color"],
            "source_record": int(r["ID"]), "shelter_id": sid, "location": {"city": "Austin", "state": "TX"}, "status": "Available",
            "description": f"{r['Breed']} ({r['Color']}), {r['AgeuponOutcome']} old. Record from Austin Animal Center open data.",
            "images": [], "views": 0, "favorites_count": 0, "application_count": 0, "created_at": now(), "updated_at": now()})
    db.pets.insert_many(docs)
    pw = generate_password_hash(os.getenv("DEMO_PASSWORD", "ChangeMe@123"))
    for n, e, role, kw in (("Demo Adopter", "demo@pawsync.com", "adopter", {"preferences": {"species": "Dog", "age_group": "Young"}}),
                           ("Shelter Manager", "shelter@pawsync.com", "shelter", {"shelter_id": sid}), ("Admin", "admin@pawsync.com", "admin", {})):
        db.users.insert_one({"name": n, "email": e, "password_hash": pw, "role": role, "location": {"city": "Austin", "state": "TX"}, "created_at": now(), **kw})
if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    from db import get_db; seed(get_db(), force=True); print("Seeded real records.")
