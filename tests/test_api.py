import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.pop("MONGO_URI", None); os.environ["SEED_PETS"] = "150"
import pytest
from app import create_app
@pytest.fixture(scope="module")
def c(): return create_app().test_client()
def tok(c, e): return c.post("/api/auth/login", json={"email": e, "password": os.getenv("DEMO_PASSWORD", "ChangeMe@123")}).json["token"]
def H(t): return {"Authorization": f"Bearer {t}"}
def test_login_and_me(c): assert c.get("/api/auth/me", headers=H(tok(c, "demo@pawsync.com"))).json["user"]["role"] == "adopter"
def test_bad_login(c): assert c.post("/api/auth/login", json={"email": "x@y.com", "password": "no"}).status_code == 401
def test_pets_and_prediction(c):
    r = c.get("/api/pets?species=Dog&limit=5").json
    assert r["items"] and all(p["species"] == "Dog" for p in r["items"])
    assert c.get(f"/api/pets/{r['items'][0]['_id']}").json["prediction"]["factors"]
def test_permissions(c): assert c.post("/api/pets", json={}, headers=H(tok(c, "demo@pawsync.com"))).status_code == 403
def test_recs_and_search(c):
    assert c.get("/api/recommendations", headers=H(tok(c, "demo@pawsync.com"))).json["items"]
    f = c.post("/api/ai/search", json={"query": "young female dog"}).json["filters"]
    assert f["species"] == "Dog" and f["sex"] == "Female" and f["age_group"] == "Young"
def test_apply_flow(c):
    t = tok(c, "demo@pawsync.com"); pid = c.get("/api/pets?limit=5").json["items"][0]["_id"]
    assert c.post("/api/applications", json={"pet_id": pid, "reason": "x"}, headers=H(t)).status_code in (201, 409)
def test_analytics_forecast_admin(c):
    assert c.get("/api/shelter/analytics", headers=H(tok(c, "shelter@pawsync.com"))).json["at_risk"]
    assert "Dog" in c.get("/api/forecast").json["forecast"]
    assert c.get("/api/admin/stats", headers=H(tok(c, "admin@pawsync.com"))).json["model"]["metrics"]["roc_auc"] > .7
