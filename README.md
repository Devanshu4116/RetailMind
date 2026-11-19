# RetailMind AI — Web Dashboard

A custom HTML/CSS/JavaScript frontend with a Flask API for the existing RetailMind AI retail forecasting and inventory project.

## Expected files

Keep your existing model and CSV outputs alongside `app.py`:

```text
retailmind/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── weekly_demand.csv
├── outputs/
│   ├── inventory_recommendations.csv
│   ├── monthly_revenue.csv
│   ├── top_products.csv
│   └── model_evaluation.csv
└── models/
    └── forecast_model.joblib
```

The app also checks for CSVs directly beside `app.py` if they are not under `outputs/`.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- The forecasting endpoint expects the saved joblib bundle to contain `model`, `feature_cols`, and optionally `product_id_map`.
- Inventory quantities, lead times and safety-stock values are simulated assumptions unless you connect real inventory data.
- The page loads Plotly.js and Google Fonts from their CDNs, so charts and fonts require internet access.
- For public deployment, use a production WSGI server and do not expose Flask debug mode.
