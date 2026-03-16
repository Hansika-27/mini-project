from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
model    = joblib.load(os.path.join(BASE_DIR, "waste_model.pkl"))

with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

METRICS  = metadata["metrics"]
RATIOS   = metadata["ratios"]   # saved by train_model_new.py

DAY_ORDER = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

# Ratios derived from training CSV
PEOPLE_RATIO    = RATIOS["people_per_student"]
NONVEG_KG_RATIO = RATIOS["nonveg_kg_per_person"]
VEG_KG_RATIO    = RATIOS["veg_kg_per_person"]

# ── CORS ───────────────────────────────────────────────────────────────────────
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.after_request
def after_request(response):
    return _cors(response)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import Response
        return _cors(Response(status=204))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse(data: dict):
    """
    Parse payload from page_new.tsx:

    Sent by page:
        date, dayOfWeek, mealType, studentsEnrolled, averageAttendance,
        specialEvent, weather, holidayPeriod, menusServed,
        leftoverFromPreviousDay, menuItems (list of strings)

    nonveg_items  → count of 'nonveg' entries in menuItems
    People_served → students × PEOPLE_RATIO
    Main_nonveg_kg / Main_veg_kg → derived from ratios
    """
    errors = []
    required = ["studentsEnrolled", "averageAttendance", "menusServed",
                "dayOfWeek", "mealType", "specialEvent"]
    for f in required:
        if f not in data:
            errors.append(f"Missing: '{f}'")
    if errors:
        return None, None, errors

    try:
        students   = float(data["studentsEnrolled"])
        attendance = float(data["averageAttendance"])
        menu_count = float(data.get("menusServed") or 0)
        leftover   = float(data.get("leftoverFromPreviousDay") or 0)
        special    = 1 if str(data.get("specialEvent","no")).lower() in ("yes","true","1") else 0
        meal_type  = 1 if str(data.get("mealType","lunch")).lower() in ("dinner","snacks") else 0
        day        = DAY_ORDER.get(str(data.get("dayOfWeek","Monday")).capitalize(), 0)

        # nonveg_items = number of 'nonveg' selections in the dynamic menu list
        menu_items = data.get("menuItems", [])
        nonveg     = float(sum(1 for item in menu_items if item == "nonveg"))

    except (TypeError, ValueError) as e:
        return None, None, [str(e)]

    # Derive missing model features from saved ratios
    people         = round(students * PEOPLE_RATIO)
    main_nonveg_kg = round(people * NONVEG_KG_RATIO, 2) if nonveg else 0.0
    main_veg_kg    = round(people * VEG_KG_RATIO, 2)

    # Feature order must match training exactly:
    # Students_enrolled, Attendance_percent, Special_event_enc, Menu_count,
    # Previous_day_leftover_kg, Nonveg_items, Meal_type_enc, Day_enc,
    # Main_nonveg_kg, Main_veg_kg, People_served
    vector = np.array([[
        students, attendance, special, menu_count, leftover,
        nonveg, meal_type, day, main_nonveg_kg, main_veg_kg, people
    ]])

    context = {
        "people":     people,
        "nonveg":     nonveg,
        "menu_count": menu_count,
        "students":   students,
        "attendance": attendance,
    }

    return vector, context, []


def _risk_level(waste_kg: float) -> str:
    """Match page_new.tsx thresholds: >50 High, >30 Medium, else Low."""
    if waste_kg > 50:
        return "High"
    elif waste_kg > 30:
        return "Medium"
    return "Low"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Food Wastage Predictor API",
        "model":   "GradientBoostingRegressor",
        "metrics": METRICS,
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify(metadata)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main endpoint consumed by page_new.tsx.

    Returns:
        predicted_waste_kg  → shown as predictedWaste
        cost                → predictedWaste × 25  (rupees)
        recommendation      → High / Medium / Low
        confidence          → model R2 x 100 (fixed, from training)
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    vector, context, errors = _parse(data)
    if errors:
        return jsonify({"errors": errors}), 422

    predicted_waste = max(0.0, round(float(model.predict(vector)[0]), 1))
    cost            = round(predicted_waste * 25)
    recommendation  = _risk_level(predicted_waste)
    confidence      = round(METRICS["r2"] * 100, 1)   # e.g. 91.5

    return jsonify({
        "predicted_waste_kg": predicted_waste,
        "cost":               cost,
        "recommendation":     recommendation,
        "confidence":         confidence,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
