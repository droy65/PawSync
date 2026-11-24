🐾 PawSync — Smarter Matches. Happier Homes.

**An intelligent animal adoption platform that connects animals with potential adopters using Machine Learning, personalized recommendations, and real-world shelter data.**

PawSync is a Flask-based animal adoption platform built using **Python, MongoDB, Machine Learning, and real Austin Animal Center data**. It combines adoption probability prediction, personalized pet recommendations, demand forecasting, and shelter priority analysis to support smarter adoption decisions.

Unlike projects that rely on artificially generated datasets, PawSync uses real historical animal shelter records.

---

✨ Key Features

- **Adoption Probability Prediction:** Predicts the likelihood of an animal being adopted using trained Machine Learning models.
- **Personalized Pet Recommendations:** Recommends animals based on adopter preferences, favorites, viewed pets, and predicted adoption likelihood.
- **Adoption Demand Forecasting:** Analyzes historical monthly adoption trends and forecasts future demand with a 95% confidence band.
- **Animal Priority Engine:** Classifies animals into Critical, High, Medium, and Low priority based on adoption likelihood and available animal attributes.
- **Intelligent Search:** Uses natural-language intent parsing to translate search queries into MongoDB queries.
- **Model Explainability:** Provides occlusion-based feature attribution to explain individual predictions.
- **Real-Time Database Integration:** Uses MongoDB through PyMongo to manage animal records and application interactions.

🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Database | MongoDB, PyMongo |
| Machine Learning | Scikit-learn |
| ML Algorithms | Random Forest, Gradient Boosting, Logistic Regression |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |

🧠 Machine Learning Implementation

PawSync uses historical animal shelter outcomes to estimate the probability of adoption.

**Dataset Overview**

- 26,729 labeled real animal outcomes used for model training and evaluation.
- 11,456 real animals with unknown outcomes used as the adoption catalogue.
- Historical records cover October 2013 to February 2016.
- 400 animals are loaded into the database by default.

**Model Training**

Three classification algorithms are compared:

1. Random Forest
2. Gradient Boosting
3. Logistic Regression

A time-based data split is used to evaluate performance:

- 80% earliest records for training.
- 20% latest records for testing.

**Model Evaluation**

| Metric | Test Performance |
|---|---:|
| ROC-AUC | ~0.83 |
| F1 Score | ~0.69 |
| PR-AUC | ~0.78 |

The model is selected based on ROC-AUC performance.

**Additional ML Components**

- Occlusion-based attribution for prediction explanations (SHAP-style, without the SHAP library).
- Cosine similarity-based recommendation engine.
- Linear trend-based monthly adoption demand forecasting.
- Rule-based animal priority classification.

📂 Project Structure

```text
PawSync/
│
├── backend/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── model.joblib
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── app.py
│   ├── db.py
│   ├── seed.py
│   └── services.py
│
├── data/
│   ├── austin_animals_unlabeled.csv
│   ├── austin_outcomes_labeled.csv
│   └── DATA_SOURCE.md
│
├── tests/
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

🚀 Getting Started

Follow these steps to run PawSync locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PawSync.git
cd PawSync
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file:

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Configure your MongoDB connection using the `.env` file.

- Leave `MONGO_URI` empty to use the in-memory demo database.
- Provide a MongoDB Atlas or local MongoDB URI to enable persistent storage.

### 4. Train the Machine Learning Models

```bash
python backend/ml/model.py
```

This trains and compares the classification models using the historical dataset.

Model training also runs automatically on the first application start when required.

### 5. Seed the Database

```bash
python backend/seed.py
```

This loads 400 real animal records into MongoDB by default.

To change the number of animals loaded, configure:

```text
SEED_PETS=N
```

Database seeding also happens automatically when the database is empty.

### 6. Start the Application

```bash
python backend/app.py
```

Open your browser and visit:

**http://localhost:5000**

🐳 Run Using Docker

PawSync also supports Docker-based deployment.

Make sure Docker Desktop is running, then execute:

```bash
docker compose up --build
```

🧪 Run Tests

Execute the following command from the project root:

```bash
pytest -q tests
```

This runs the project's API tests using Pytest.

🔐 Demo Login Credentials

The application provides demo accounts for exploring different user roles.

| Account | Email |
|---|---|
| Demo User | demo@pawsync.com |
| Shelter | shelter@pawsync.com |
| Admin | admin@pawsync.com |

**Password:** `DEMO_PASSWORD`

The password is configured through the application's environment settings.

📊 Data Source & Transparency

PawSync uses real animal shelter data from the **Austin Animal Center**.

The dataset contains historical adoption outcomes and animal records. Details about the original data source are available in:

`data/DATA_SOURCE.md`

**Important Data Limitations**

The original dataset does not contain the following information:

- Animal size
- Temperament
- Health and vaccination details
- Adoption fees
- Days spent in shelter
- Shelter capacity or overcrowding
- Animal photographs

These fields have deliberately been excluded rather than filled with fabricated information.

Adopters, applications, favorites, and user interactions are generated through actual use of the application, not through seeded synthetic records.

⚠️ Current Scope & Limitations

The following features are not currently implemented:

- XGBoost
- SHAP library integration
- Messaging functionality
- Donations and events
- PDF/CSV report generation
- Jupyter notebooks
- Rate limiting
- AWS deployment scripts

The demand forecasting component uses approximately 28 months of historical data, so its forecasts should be interpreted with that limitation in mind.

🎯 Project Objective

PawSync aims to demonstrate how Machine Learning and data-driven decision-making can be applied to animal adoption and shelter management.

By combining adoption prediction, personalized recommendations, demand forecasting, and animal prioritization, the platform explores ways to make the adoption process more informed and efficient.

---

**PawSync — Smarter Matches. Happier Homes. 🐾**
:::
