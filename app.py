from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATHS = [BASE_DIR / "models" / "forecast_model.joblib", BASE_DIR / "forecast_model.joblib"]

app = Flask(__name__)

def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return None

def load_csv(filename, alternatives=()):
    path = first_existing(*(OUTPUT_DIR / filename, DATA_DIR / filename),
                          *(OUTPUT_DIR / x for x in alternatives),
                          *(DATA_DIR / x for x in alternatives))
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

weekly = load_csv("weekly_demand.csv")
inventory = load_csv("inventory_recommendations.csv")
monthly = load_csv("monthly_revenue.csv")
top_products = load_csv("top_products.csv")
evaluation = load_csv("model_evaluation.csv")

bundle = {}
model = None
feature_cols = ["product_id", "lag_1", "lag_2", "lag_4", "lag_8",
                "rolling_mean_4", "rolling_std_4", "month", "week_of_year"]
product_id_map = {}
for model_path in MODEL_PATHS:
    if model_path.exists():
        try:
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict):
                bundle = loaded
                model = loaded.get("model")
                feature_cols = loaded.get("feature_cols", feature_cols)
                product_id_map = loaded.get("product_id_map", {})
            else:
                model = loaded
            break
        except Exception as exc:
            print(f"Could not load model {model_path}: {exc}")

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

weekly = normalize_columns(weekly)
inventory = normalize_columns(inventory)
monthly = normalize_columns(monthly)
top_products = normalize_columns(top_products)
evaluation = normalize_columns(evaluation)

if not weekly.empty:
    if "Week" not in weekly.columns:
        for candidate in ("InvoiceDate", "week", "date", "Date"):
            if candidate in weekly.columns:
                weekly = weekly.rename(columns={candidate: "Week"})
                break
    if "Demand" not in weekly.columns:
        for candidate in ("Quantity", "quantity", "weekly_demand"):
            if candidate in weekly.columns:
                weekly = weekly.rename(columns={candidate: "Demand"})
                break
    if "StockCode" in weekly.columns:
        weekly["StockCode"] = weekly["StockCode"].astype(str)
    if "Week" in weekly.columns:
        weekly["Week"] = pd.to_datetime(weekly["Week"], errors="coerce")
    if "Demand" in weekly.columns:
        weekly["Demand"] = pd.to_numeric(weekly["Demand"], errors="coerce").fillna(0)

def num_col(df, candidates, default=0):
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)

def first_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)

def records_clean(df):
    if df.empty:
        return []
    out = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return out.to_dict(orient="records")

def product_list():
    values = set()
    if not weekly.empty and "StockCode" in weekly.columns:
        values.update(weekly["StockCode"].dropna().astype(str).unique().tolist())
    if not inventory.empty and "StockCode" in inventory.columns:
        values.update(inventory["StockCode"].dropna().astype(str).unique().tolist())
    return sorted(values)


def inventory_view():
    df = inventory.copy()

    if df.empty:
        return df

    # Normalize column names regardless of capitalization.
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map the original CSV columns to the names used by the dashboard.
    mapping = {
        "stockcode": ["stockcode", "stock_code", "product_id"],
        "avg_weekly_demand": [
            "avg_weekly_demand", "average_weekly_demand"
        ],
        "current_stock": [
            "current_stock", "simulated_stock"
        ],
        "safety_stock": ["safety_stock"],
        "reorder_point": ["reorder_point"],
        "recommended_order": [
            "recommended_order", "recommended_order_qty"
        ],
        "status": [
            "status", "stock_status", "inventory_status"
        ],
    }

    for target, candidates in mapping.items():
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[target] = df[candidate]
                    break

    if "stockcode" not in df.columns:
        df["stockcode"] = [
            f"Item {i + 1}" for i in range(len(df))
        ]

    df = df.rename(columns={"stockcode": "StockCode"})

    numeric_columns = [
        "avg_weekly_demand",
        "current_stock",
        "safety_stock",
        "reorder_point",
        "recommended_order",
    ]

    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    if "status" not in df.columns:
        df["status"] = np.where(
            df["recommended_order"] > 0,
            "Reorder Needed",
            "Healthy"
        )

    df["status"] = df["status"].fillna("Unknown").astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    return df


def weekly_history(code):
    if weekly.empty or not {"StockCode", "Week", "Demand"}.issubset(weekly.columns):
        return pd.DataFrame(columns=["Week", "Demand"])
    hist = weekly[weekly["StockCode"].astype(str) == str(code)][["Week", "Demand"]].dropna().sort_values("Week")
    return hist

def get_product_id(code):
    # Saved mapping may be StockCode -> integer ID or integer ID -> StockCode.
    if isinstance(product_id_map, dict):
        if str(code) in product_id_map:
            try:
                return int(product_id_map[str(code)])
            except Exception:
                pass
        for key, value in product_id_map.items():
            if str(value) == str(code):
                try:
                    return int(key)
                except Exception:
                    pass
    codes = product_list()
    try:
        return codes.index(str(code))
    except ValueError:
        return 0

def forecast_for(code, horizon=4):
    hist = weekly_history(code)
    if hist.empty:
        raise ValueError("No weekly history is available for this product.")
    if model is None:
        raise ValueError("The trained forecast model was not found. Check models/forecast_model.joblib.")
    demand = [max(0.0, float(x)) for x in hist["Demand"].fillna(0).tolist()]
    dates = pd.to_datetime(hist["Week"]).tolist()
    last_date = pd.Timestamp(dates[-1])
    output = []
    pid = get_product_id(str(code))
    for step in range(1, horizon + 1):
        last = demand[-1]
        def lag(k):
            return demand[-k] if len(demand) >= k else 0.0
        recent4 = demand[-4:]
        recent_std = float(np.std(recent4, ddof=1)) if len(recent4) > 1 else 0.0
        next_date = last_date + pd.Timedelta(days=7)
        iso_week = int(next_date.isocalendar().week)
        features = {
            "product_id": pid,
            "lag_1": lag(1),
            "lag_2": lag(2),
            "lag_4": lag(4),
            "lag_8": lag(8),
            "rolling_mean_4": float(np.mean(recent4)) if recent4 else 0.0,
            "rolling_std_4": recent_std,
            "month": int(next_date.month),
            "week_of_year": iso_week
        }
        X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}], columns=feature_cols)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)
        output.append({"week": next_date.strftime("%Y-%m-%d"), "demand": round(pred, 2), "step": step})
        demand.append(pred)
        last_date = next_date
    return {
        "product": str(code),
        "history": [{"week": pd.Timestamp(r.Week).strftime("%Y-%m-%d"), "demand": round(float(r.Demand), 2)}
                    for r in hist.tail(40).itertuples()],
        "forecast": output
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/overview")
def api_overview():
    inv = inventory_view()
    monthly_df = monthly.copy()
    if not monthly_df.empty:
        if len(monthly_df.columns) >= 2:
            monthly_df = monthly_df.iloc[:, :2].copy()
            monthly_df.columns = ["month", "revenue"]
            monthly_df["revenue"] = pd.to_numeric(monthly_df["revenue"], errors="coerce").fillna(0)
            monthly_df["month"] = monthly_df["month"].astype(str)
    top_df = top_products.copy()
    if not top_df.empty and len(top_df.columns) >= 2:
        top_df = top_df.iloc[:, :2].copy()
        top_df.columns = ["product", "revenue"]
        top_df["revenue"] = pd.to_numeric(top_df["revenue"], errors="coerce").fillna(0)
        top_df = top_df.sort_values("revenue", ascending=False).head(10)
    status_counts = {}
    if not inv.empty:
        status_counts = inv["status"].value_counts().to_dict()
    total_revenue = float(pd.to_numeric(monthly_df.get("revenue", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_products = len(product_list())
    metrics = {
        "total_revenue": round(total_revenue, 2),
        "products_tracked": total_products,
        "reorder_alerts": int((inv["status"].str.lower().str.contains("reorder")).sum()) if not inv.empty else 0,
        "potential_overstock": int((inv["status"].str.lower().str.contains("overstock")).sum()) if not inv.empty else 0
    }
    return jsonify({"metrics": metrics, "monthly": records_clean(monthly_df), "top_products": records_clean(top_df),
                    "status_counts": [{"status": k, "count": int(v)} for k, v in status_counts.items()],
                    "evaluation": records_clean(evaluation)})

@app.route("/api/products")
def api_products():
    return jsonify({"products": product_list()})

@app.route("/api/forecast")
def api_forecast():
    code = request.args.get("product", "")
    try:
        horizon = max(1, min(4, int(request.args.get("horizon", "4"))))
    except ValueError:
        horizon = 4
    if not code:
        return jsonify({"error": "Choose a product first."}), 400
    try:
        return jsonify(forecast_for(code, horizon))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/inventory")
def api_inventory():
    df = inventory_view()

    status = request.args.get("status", "All")
    search = request.args.get("search", "").strip().lower()

    if not df.empty:
        if status != "All":
            df = df[
                df["status"].str.lower() == status.lower()
            ]

        if search:
            df = df[
                df["StockCode"].str.lower().str.contains(
                    search, na=False
                )
            ]

        cols = [
            "StockCode",
            "avg_weekly_demand",
            "current_stock",
            "safety_stock",
            "reorder_point",
            "recommended_order",
            "status",
        ]

        return jsonify({"rows": records_clean(df[cols])})

    return jsonify({"rows": []})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    payload = request.get_json(silent=True) or {}
    try:
        avg_demand = max(0.0, float(payload.get("avg_demand", 10)))
        demand_std = max(0.0, float(payload.get("demand_std", 2)))
        current_stock = max(0.0, float(payload.get("current_stock", 20)))
        lead_time = max(1.0, min(12.0, float(payload.get("lead_time", 2))))
        safety_factor = max(0.0, min(4.0, float(payload.get("safety_factor", 1.65))))
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter valid numeric values."}), 400
    safety_stock = math.ceil(safety_factor * demand_std * math.sqrt(lead_time))
    reorder_point = math.ceil(avg_demand * lead_time + safety_stock)
    order_qty = max(0, math.ceil(reorder_point - current_stock))
    status = "Reorder Needed" if order_qty > 0 else ("Potential Overstock" if current_stock > avg_demand * lead_time * 3 else "Healthy")
    return jsonify({"safety_stock": safety_stock, "reorder_point": reorder_point, "recommended_order": order_qty, "status": status})

@app.route("/api/inventory.csv")
def inventory_csv():
    from flask import Response
    df = inventory_view()
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=retailmind_inventory.csv"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
