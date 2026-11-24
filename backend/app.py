"""PAWSYNC Flask API. Run: python backend/app.py"""
import os, logging, datetime as dt
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt, verify_jwt_in_request
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db
import services as sv
from ml import model as mlm
from seed import seed
logging.basicConfig(level=logging.INFO)
def create_app():
    app = Flask(__name__)
    app.config["JWT_SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me-please-32-bytes!!")
    CORS(app); JWTManager(app); db = get_db(fresh=True); seed(db)
    def err(msg, code, status): return jsonify(success=False, message=msg, error_code=code), status
    def oid(s):
        try: return ObjectId(s)
        except (InvalidId, TypeError): return None
    def me(): return db.users.find_one({"_id": oid(get_jwt_identity())})
    def opt_user():
        try:
            verify_jwt_in_request(optional=True); return oid(get_jwt_identity()) if get_jwt_identity() else None
        except Exception: return None
    def roles(*rs):
        def deco(f):
            @jwt_required()
            def w(*a, **k):
                if get_jwt().get("role") not in rs: return err("Forbidden", "FORBIDDEN", 403)
                return f(*a, **k)
            w.__name__ = f.__name__; return w
        return deco
    @app.errorhandler(404)
    def _404(e): return err("Not found", "NOT_FOUND", 404)
    @app.errorhandler(Exception)
    def _500(e):
        if hasattr(e, "code") and isinstance(e.code, int): return err(getattr(e, "description", "Error"), "HTTP_ERROR", e.code)
        logging.exception(e); return err("Internal server error", "SERVER_ERROR", 500)
    @app.get("/")
    def index(): return render_template("index.html")
    # ---- auth
    @app.post("/api/auth/register")
    def register():
        d = request.get_json(silent=True) or {}
        if not d.get("email") or "@" not in d["email"] or len(d.get("password", "")) < 8 or not d.get("name"):
            return err("Name, valid email and 8+ char password required", "VALIDATION", 400)
        if db.users.find_one({"email": d["email"].lower()}): return err("Email already registered", "EMAIL_EXISTS", 409)
        u = {"name": d["name"], "email": d["email"].lower(), "password_hash": generate_password_hash(d["password"]), "role": "adopter",
             "location": {"city": d.get("city", "Jalandhar"), "state": "Punjab"}, "preferences": d.get("preferences", {}), "created_at": sv.now()}
        u["_id"] = db.users.insert_one(u).inserted_id
        return jsonify(token=create_access_token(str(u["_id"]), additional_claims={"role": "adopter"}), user=sv.ser(u)), 201
    @app.post("/api/auth/login")
    def login():
        d = request.get_json(silent=True) or {}
        u = db.users.find_one({"email": str(d.get("email", "")).lower()})
        if not u or u.get("suspended") or not check_password_hash(u["password_hash"], d.get("password", "")):
            return err("Invalid credentials", "BAD_LOGIN", 401)
        return jsonify(token=create_access_token(str(u["_id"]), additional_claims={"role": u["role"]}), user=sv.ser(u))
    @app.get("/api/auth/me")
    @jwt_required()
    def whoami(): return jsonify(user=sv.ser(me()))
    @app.put("/api/auth/preferences")
    @jwt_required()
    def prefs():
        db.users.update_one({"_id": oid(get_jwt_identity())}, {"$set": {"preferences": request.get_json() or {}}}); return jsonify(success=True)
    # ---- pets
    def card(p):
        pr = mlm.predict(p, explain=False)
        s = db.shelters.find_one({"_id": p["shelter_id"]}) or {}
        return dict(sv.ser(p), prediction=pr, shelter_name=s.get("name"), shelter_verified=s.get("verified"))
    @app.get("/api/pets")
    def pets():
        a = request.args; q = sv.build_query(a.to_dict())
        if a.get("status"): q["status"] = a["status"]
        if a.get("shelter_id") and oid(a["shelter_id"]): q["shelter_id"] = oid(a["shelter_id"])
        sort = {"views": ("views", -1), "youngest": ("age_months", 1), "oldest": ("age_months", -1)}.get(a.get("sort"), ("created_at", -1))
        page, limit = max(int(a.get("page", 1)), 1), min(int(a.get("limit", 12)), 50)
        total = db.pets.count_documents(q)
        items = list(db.pets.find(q).sort(*sort).skip((page - 1) * limit).limit(limit))
        return jsonify(items=[card(p) for p in items], total=total, page=page, pages=-(-total // limit))
    @app.get("/api/pets/<pid>")
    def pet(pid):
        p = db.pets.find_one({"_id": oid(pid)})
        if not p: return err("Pet not found", "PET_NOT_FOUND", 404)
        db.pets.update_one({"_id": p["_id"]}, {"$inc": {"views": 1}})
        pr = mlm.predict(p)
        db.predictions.update_one({"pet_id": p["_id"]}, {"$set": dict(pr, pet_id=p["_id"], at=sv.now())}, upsert=True)
        uid = opt_user()
        if uid: db.interactions.insert_one({"user_id": uid, "pet_id": p["_id"], "event": "pet_view", "timestamp": sv.now()})
        c = card(p); c["prediction"] = pr
        c["insight"] = (f"Based on {mlm.bundle()['n_train']:,} real Austin Animal Center outcomes, animals like {p['name']} are adopted "
                        f"{round(pr['probability']*100)}% of the time (shelter-wide average {round(pr['baseline']*100)}%). Biggest factor: {pr['factors'][0]['feature']}.")
        return jsonify(c)
    @app.post("/api/pets")
    @roles("shelter", "admin")
    def add_pet():
        d = request.get_json(silent=True) or {}; u = me()
        need = ("name", "species", "breed", "age_months", "sex")
        if any(k not in d for k in need) or d["species"] not in ("Dog", "Cat"): return err("Required: " + ", ".join(need) + " (species Dog/Cat)", "VALIDATION", 400)
        sid = u.get("shelter_id") or oid(d.get("shelter_id")); s = db.shelters.find_one({"_id": sid}) or {}
        p = {"name": d["name"], "has_name": True, "species": d["species"], "breed": d["breed"], "is_mix": "mix" in d["breed"].lower(), "age_months": float(d["age_months"]),
             "sex": d["sex"], "altered": d.get("altered"), "color": d.get("color", "Unknown"), "shelter_id": sid, "location": {"city": s.get("city", "Austin"), "state": s.get("state", "TX")},
             "status": "Available", "images": [], "views": 0, "favorites_count": 0, "application_count": 0, "description": d.get("description", ""), "created_at": sv.now(), "updated_at": sv.now()}
        p["_id"] = db.pets.insert_one(p).inserted_id
        return jsonify(sv.ser(p)), 201
    @app.delete("/api/pets/<pid>")
    @roles("shelter", "admin")
    def del_pet(pid):
        p = db.pets.find_one({"_id": oid(pid)}); u = me()
        if not p: return err("Pet not found", "PET_NOT_FOUND", 404)
        if u["role"] == "shelter" and p["shelter_id"] != u.get("shelter_id"): return err("Forbidden", "FORBIDDEN", 403)
        db.pets.delete_one({"_id": p["_id"]}); return jsonify(success=True)
    # ---- favorites / recommendations / AI
    @app.post("/api/favorites/<pid>")
    @jwt_required()
    def fav(pid):
        k = {"user_id": oid(get_jwt_identity()), "pet_id": oid(pid)}
        if db.favorites.find_one(k):
            db.favorites.delete_one(k); db.pets.update_one({"_id": k["pet_id"]}, {"$inc": {"favorites_count": -1}}); return jsonify(favorited=False)
        db.favorites.insert_one(dict(k, created_at=sv.now())); db.pets.update_one({"_id": k["pet_id"]}, {"$inc": {"favorites_count": 1}})
        return jsonify(favorited=True)
    @app.get("/api/favorites")
    @jwt_required()
    def favs():
        ids = [f["pet_id"] for f in db.favorites.find({"user_id": oid(get_jwt_identity())})]
        return jsonify(items=[card(p) for p in db.pets.find({"_id": {"$in": ids}})])
    @app.get("/api/recommendations")
    @jwt_required()
    def recs(): return jsonify(items=sv.recommend(db, me()))
    @app.post("/api/ai/search")
    def ai_search():
        text = (request.get_json(silent=True) or {}).get("query", "").strip()
        if not text: return err("Query required", "VALIDATION", 400)
        f = sv.extract_intent(text, db.pets.distinct("breed")); items = [card(p) for p in db.pets.find(sv.build_query(f)).limit(8)]
        uid = opt_user()
        if uid: db.interactions.insert_one({"user_id": uid, "event": "search", "query": text, "filters": f, "timestamp": sv.now()})
        msg = (f"I found {len(items)} available pet(s) matching your request." if items
               else "I don't have enough information to answer that accurately - no available pets match those filters.")
        return jsonify(filters=f, items=items, message=msg)
    # ---- applications
    @app.post("/api/applications")
    @roles("adopter")
    def apply():
        d = request.get_json(silent=True) or {}; p = db.pets.find_one({"_id": oid(d.get("pet_id"))})
        if not p or p["status"] != "Available": return err("Pet not available", "PET_UNAVAILABLE", 400)
        uid = oid(get_jwt_identity())
        if db.applications.find_one({"adopter_id": uid, "pet_id": p["_id"], "status": {"$nin": ["Rejected", "Withdrawn"]}}):
            return err("You already applied for this pet", "DUPLICATE", 409)
        a = {"adopter_id": uid, "pet_id": p["_id"], "shelter_id": p["shelter_id"], "reason": d.get("reason", ""), "status": "Pending",
             "application_date": sv.now(), "history": [{"status": "Pending", "at": sv.now(), "note": "Submitted"}]}
        a["_id"] = db.applications.insert_one(a).inserted_id
        db.pets.update_one({"_id": p["_id"]}, {"$inc": {"application_count": 1}}); return jsonify(sv.ser(a)), 201
    @app.get("/api/applications")
    @jwt_required()
    def my_apps():
        u = me(); q = {"adopter_id": u["_id"]} if u["role"] == "adopter" else {"shelter_id": u.get("shelter_id")} if u["role"] == "shelter" else {}
        out = []
        for a in db.applications.find(q).sort("application_date", -1).limit(100):
            p = db.pets.find_one({"_id": a["pet_id"]}) or {}; ad = db.users.find_one({"_id": a["adopter_id"]}) or {}
            out.append(dict(sv.ser(a), pet_name=p.get("name"), adopter_name=ad.get("name")))
        return jsonify(items=out)
    @app.put("/api/applications/<aid>")
    @roles("shelter", "admin")
    def review(aid):
        st = (request.get_json(silent=True) or {}).get("status")
        if st not in ("Under Review", "Interview", "Approved", "Rejected", "Adopted"): return err("Invalid status", "VALIDATION", 400)
        a = db.applications.find_one({"_id": oid(aid)}); u = me()
        if not a: return err("Not found", "NOT_FOUND", 404)
        if u["role"] == "shelter" and a["shelter_id"] != u.get("shelter_id"): return err("Forbidden", "FORBIDDEN", 403)
        db.applications.update_one({"_id": a["_id"]}, {"$set": {"status": st, "reviewed_by": u["_id"], "reviewed_at": sv.now()},
                                   "$push": {"history": {"status": st, "at": sv.now()}}})
        if st == "Approved": db.pets.update_one({"_id": a["pet_id"]}, {"$set": {"status": "Pending"}})
        if st == "Adopted":
            db.pets.update_one({"_id": a["pet_id"]}, {"$set": {"status": "Adopted", "adopted_at": sv.now()}})
            db.adoptions.insert_one({"application_id": a["_id"], "pet_id": a["pet_id"], "adopter_id": a["adopter_id"], "date": sv.now()})
        db.notifications.insert_one({"user_id": a["adopter_id"], "message": f"Your application is now: {st}", "created_at": sv.now(), "read": False})
        return jsonify(success=True)
    @app.get("/api/notifications")
    @jwt_required()
    def notes(): return jsonify(items=sv.ser(list(db.notifications.find({"user_id": oid(get_jwt_identity())}).sort("created_at", -1).limit(20))))
    # ---- analytics
    _stats = {}
    def dstats():
        if not _stats: _stats.update(mlm.stats())
        return _stats
    @app.get("/api/forecast")
    def forecast(): return jsonify(months=dstats()["months"], forecast=dstats()["forecast"], note="Monthly adoption records in the real Austin dataset; linear trend with 95% band.")
    @app.get("/api/shelter/analytics")
    @roles("shelter", "admin")
    def analytics():
        u = me(); sid = u.get("shelter_id") or oid(request.args.get("shelter_id")); base = {"shelter_id": sid}
        grp = lambda coll, f: {r["_id"]: r["n"] for r in coll.aggregate([{"$match": base}, {"$group": {"_id": f, "n": {"$sum": 1}}}])}
        by_status, by_species, apps = grp(db.pets, "$status"), grp(db.pets, "$species"), grp(db.applications, "$status")
        pets = list(db.pets.find(dict(base, status="Available"))); probs = mlm.predict_many(pets); risk = []
        for p, pb in zip(pets, probs):
            sc, lvl = sv.priority(p, pb); risk.append({"id": str(p["_id"]), "name": p["name"], "species": p["species"], "breed": p["breed"], "age_months": p["age_months"], "probability": round(float(pb), 3), "score": sc, "level": lvl})
        risk.sort(key=lambda r: -r["score"]); lv = {k: sum(1 for r in risk if r["level"] == k) for k in ("Critical", "High", "Medium", "Low")}
        ds = dstats(); ag = ds["adoption_rate_by_age"]; tips = []
        if lv["Critical"] + lv["High"]: tips.append(f"{lv['Critical'] + lv['High']} listed pets are Critical/High priority - feature them first.")
        if ag: tips.append(f"Historical data: animals under 1 year are adopted {round(ag.get('<1 yr', 0)*100)}% of the time vs {round(ag.get('7+ yrs', 0)*100)}% for 7+ years - promote seniors.")
        io = ds["adoption_rate_by_altered"]; tips.append(f"Spayed/neutered animals are adopted {round(io.get('Spayed/neutered', 0)*100)}% of the time vs {round(io.get('Intact', 0)*100)}% for intact ones.")
        return jsonify(shelter=sv.ser(db.shelters.find_one({"_id": sid}) or {}), by_status=by_status, by_species=by_species, applications=apps, levels=lv,
                       avg_probability=round(float(sum(probs) / max(len(probs), 1)), 3), at_risk=risk[:10], insights=tips, dataset=ds)
    @app.get("/api/admin/stats")
    @roles("admin")
    def admin_stats():
        b = mlm.bundle(); r = lambda m: {k: round(v, 3) for k, v in m.items()}
        return jsonify(users=db.users.count_documents({}), shelters=db.shelters.count_documents({}), pets=db.pets.count_documents({}),
                       applications=db.applications.count_documents({}), adoptions=db.pets.count_documents({"status": "Adopted"}),
                       model={"name": b["name"], "metrics": r(b["metrics"]), "all": {k: r(v) for k, v in b["all"].items()}, "training_records": b["n_train"], "test_records": b["n_test"],
                              "predictions": db.predictions.count_documents({}), "source": "Austin Animal Center (real), time-based split"})
    @app.get("/api/shelters")
    def shelters(): return jsonify(items=sv.ser(list(db.shelters.find())))
    return app
if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG") == "1")
