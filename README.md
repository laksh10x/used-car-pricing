# CarSight AI

CarSight AI is a decision-support tool for used car pricing. The app combines a market-based price prediction model with a damage-text analysis layer so the user gets a price range, a damage impact summary, and a plain-language explanation instead of a single number.

## What the app does

- predicts a base used-car value from structured vehicle features
- reads the seller's damage description and extracts issue categories
- estimates severity and price impact from the text
- returns an adjusted price range instead of a single price
- includes a simple chatbot for explainability and negotiation guidance

## Project structure

- `backend/` - FastAPI API, model training script, NLP logic, and tests
- `frontend/` - Vite TypeScript dashboard
- `models/` - trained model artifact and metadata
- `data/` - cached training data and processed preview files
- `assets/` - visual assets used in the interface
- `docs/` - final presentation, planning files, and report references

## Dataset and model

The pricing model was trained from the public Kaggle dataset `andreinovikov/used-cars-dataset`.

Saved evaluation metrics from the latest training run:

- MAE: `$2,939.93`
- RMSE: `$4,966.22`
- MAPE: `10.02%`
- training rows: `80,000`

The final model is a weighted hybrid blend:

- `60%` TF-IDF + Ridge regression over combined vehicle specification text
- `40%` HistGradientBoosting over structured vehicle features

This was chosen after comparing four candidates:

1. original Ridge baseline using engine text only
2. expanded Ridge NLP model using combined vehicle spec text
3. structured HistGradientBoosting model
4. weighted hybrid blend

The hybrid model gave the lowest held-out error, so it became the final pricing model. Damage text is handled separately through a rule-based NLP layer so the output stays interpretable and easy to explain in the dashboard.

## How to run the project

### 1. Backend

From `backend/`:

```powershell
$env:PYTHONPATH="C:\Users\laksh\OneDrive\Desktop\DS440\CarSight_App\backend"
C:\Users\laksh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend

From `frontend/`:

```powershell
npm install
npm run dev -- --host 127.0.0.1 --port 4173
```

Then open:

- frontend: `http://127.0.0.1:4173`
- backend health check: `http://127.0.0.1:8000/health`

## How to retrain the model

From `backend/`:

```powershell
C:\Users\laksh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe train_model.py
```

This script downloads the Kaggle dataset, cleans the data, trains the regression pipeline, evaluates the model, and saves new artifacts to `models/`.

## Testing

Backend tests:

```powershell
$env:PYTHONPATH="C:\Users\laksh\OneDrive\Desktop\DS440\CarSight_App\backend"
C:\Users\laksh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pytest -q
```

Frontend build check:

```powershell
npm run build
```

## Notes

- The app is a working capstone prototype, not a production marketplace system.
- The price range is meant to support decision-making, not replace a mechanic inspection.
- The NLP layer is intentionally simple and interpretable to match the scope of the project.

## Included project documents

- final presentation deck
- project brief
- results and conclusion reference PDF
- updated Gantt chart and critical path analysis
