# Used Car Pricing (CarSight AI)

This repository contains my IST 440 capstone project, CarSight AI. The goal of the project is to give buyers a better way to judge used car prices when a seller description is vague, incomplete, or written to downplay damage.

Most pricing tools look at structured fields like year, mileage, and model. This project adds an NLP layer so the app can also read damage-related text such as "engine knock," "small dent," or "rear bumper cracked" and fold that into the estimate.

Instead of returning one number, the app gives:

- a fair price range
- a deal meter
- detected damage categories and estimated impact
- a short explanation of why the estimate moved
- a chatbot sidebar for follow-up questions

## Project summary

The final version uses a hybrid pricing model:

- `60%` TF-IDF + Ridge regression over combined vehicle specification text
- `40%` HistGradientBoosting over structured vehicle features

The damage description is handled by a separate NLP layer that:

- cleans the text
- looks for damage-related keywords
- groups issues into categories such as body, engine, interior, tire, or electrical
- estimates severity
- applies an interpretable price adjustment

Latest saved model metrics:

- MAE: `$2,939.93`
- RMSE: `$4,966.22`
- MAPE: `10.02%`

## Repo layout

```text
used-car-pricing/
|-- assets/                  interface images
|-- backend/                 FastAPI app, training code, tests
|-- data/                    processed preview data
|-- docs/                    presentation, planning files, project references
|-- frontend/                Vite + TypeScript dashboard
|-- models/                  saved model artifact and metadata
|-- .gitignore
`-- README.md
```

## What you need before you start

- Python 3.10 or newer
- Node.js 18 or newer
- npm

This app was built and tested on Windows with PowerShell, so the commands below use that format.

## How to run the app locally

The trained model file is already included in the repo, so you do not need to retrain anything before the first run.

### 1. Clone the repository

```powershell
git clone https://github.com/laksh10x/used-car-pricing.git
cd used-car-pricing
```

### 2. Set up the backend

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the backend requirements:

```powershell
pip install -r backend/requirements.txt
```

Start the FastAPI server from the repo root:

```powershell
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If the backend started correctly, this health route should work in the browser:

```text
http://127.0.0.1:8000/health
```

### 3. Set up the frontend

Open a second terminal in the same project folder:

```powershell
cd used-car-pricing
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 4173
```

### 4. Open the app

Once both servers are running, open:

```text
http://127.0.0.1:4173
```

From there, enter the vehicle details, paste the seller's wording into the damage description box, and run the estimate.

## How to retrain the model

Retraining is optional. The repo already includes a saved model in `models/`.

If you want to retrain it from the Kaggle dataset used for the project:

1. Make sure your Kaggle access is set up on your machine.
2. Activate the same Python environment used for the backend.
3. Run:

```powershell
python backend/train_model.py
```

That script downloads the dataset, cleans the data, compares model candidates, and saves the updated artifacts back into `models/`.

## Testing

Backend tests:

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests -q
```

Frontend build check:

```powershell
cd frontend
npm run build
```
