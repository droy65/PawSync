# 🐾 PAWSYNC — Smarter Matches. Happier Homes.
Flask + PyMongo adoption platform built on **real Austin Animal Center data** (see `data/DATA_SOURCE.md`). No synthetic data.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env          # MONGO_URI empty = in-memory demo DB; set your MongoDB Atlas/local URI to persist
python backend/ml/model.py    # trains & compares models on the real data (auto-runs on first start)
python backend/seed.py        # loads 400 real animals into MongoDB (auto-runs if DB empty; SEED_PETS=N to change)
python backend/app.py         # http://localhost:5000
pytest -q tests
```
Docker: `docker compose up --build`. Demo logins: demo@/shelter@/admin@pawsync.com, password = `DEMO_PASSWORD`.

## Data & ML (honest summary)
- **Data**: 26,729 labeled real outcomes (Oct 2013–Feb 2016) train the model; 11,456 real animals with unknown outcomes are the catalogue (400 loaded by default).
- **Model**: predicts P(adopted | species, breed, age, sex, spayed/neutered, coat color, named, mixed). Random Forest / Gradient Boosting / Logistic Regression compared with a **time-based split** (train on earliest 80%, test on latest 20%). Selected by ROC-AUC (~0.83 test; F1 ~0.69; PR-AUC ~0.78).
- **Explanations**: occlusion attribution per prediction (SHAP-style; SHAP library not used).
- **Recommender**: cosine similarity of preferences + favorites + viewed pets, blended with model likelihood.
- **Demand forecast**: linear trend on real monthly adoption counts with 95% band (only ~28 months of data).
- **Priority engine**: low likelihood + older age + intact + unnamed → Critical/High/Medium/Low.
- **NL search**: rule-based intent parser → MongoDB query (replace `extract_intent` with an LLM if desired).

## Not available because the dataset doesn't contain it
Size, temperament, health/vaccination, fees, days-in-shelter, capacity/overcrowding, photos. These were removed rather than invented.
Adopters, applications, favorites and interactions are created by real use of the app, not seeded.
Not built: XGBoost, SHAP lib, messaging, donations/events, PDF/CSV reports, notebooks, rate limiting, AWS scripts.
