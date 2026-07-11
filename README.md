# MLOps Final Project
End-to-end MLOps project for training, promoting, deploying, and monitoring a house price index prediction service.

## Project Architecture

```mermaid
flowchart LR
	subgraph Data
	DVC[(DVC-tracked raw and processed data)]
	end

	subgraph Training
	T[train.py] --> M[MLflow Model Registry]
	DVC --> T
	end

	subgraph Promotion
	P[promote.py] --> M
	P -->|moves best Staging model| PROD[(Production stage)]
	end

	subgraph Deployment
	GH[GitHub Actions] --> RENDER[Render production backend]
	PROD --> RENDER
	end

	subgraph Serving
	API["Flask backend / backend.api"]
	API --> PRED["/predict"]
	API --> HEALTH["/health"]
	API --> METRICS["/metrics"]
	end

	subgraph Monitoring
	PROM[Prometheus]
	GRAF[Grafana dashboard]
	end

	RENDER --> API
	API --> PROM
	PROM --> GRAF
```

The backend serves predictions from the Production model in MLflow. The same service exposes `/health` and `/metrics`, so the live production deployment can be monitored directly.

The application also keeps a separate user-account database for Google-authenticated users. That database is independent from the MLflow/DagsHub model registry and is populated from `/auth/google` with a minimal `users` table: `email`, `name`, `created_at`.

## Repository Layout

- `backend/`: Flask API, preprocessing, and model loading helpers.
- `train.py`: trains the candidate model and registers it in MLflow as `Staging`.
- `promote.py`: compares the Staging model with the current Production model and promotes only if it performs better.
- `frontend/`: Vite-based UI for submitting predictions and viewing results.
- `monitoring/`: Prometheus and Grafana provisioning for production monitoring.
- `tests/`: unit, integration, and E2E test suites.

The app database is configured separately from MLflow with `APP_DATABASE_URL` (preferred) or `DATABASE_URL`. For Postgres, use a connection string such as `postgresql://user:password@host:5432/appdb`.

The backend database schema is defined explicitly in [backend/sql/001_create_users.sql](backend/sql/001_create_users.sql). The application applies this migration when `/auth/google` first needs the `users` table.

## How It Works

### CI/CD Pipeline

The repository uses GitHub Actions to move code and models through the environment chain.

- [`.github/workflows/dev-staging.yml`](.github/workflows/dev-staging.yml) runs on pushes to `staging`.
- It installs dependencies, runs the unit and integration test suites, trains a candidate model, deploys the backend to staging, and then runs E2E tests against the live staging environment.
- [`.github/workflows/staging-main.yml`](.github/workflows/staging-main.yml) runs on pushes to `main`.
- It repeats the model promotion gate, deploys the backend to production, then smoke-tests the production `/health` endpoint and verifies the Production model is loaded.

In short: `dev -> staging` validates the code and trains a candidate model, then `staging -> main` promotes only a model that passes the quality gate and deploys it to production.

### Model Promotion

Model promotion is handled explicitly by `promote.py`, not automatically during training.

1. `train.py` logs a new MLflow model version and assigns it to the `Staging` stage.
2. `promote.py` loads the latest Staging model and the current Production model.
3. Both models are evaluated on the supplied real-world dataset using MAE.
4. If the Staging model is better than Production, `promote.py` transitions it to `Production` and archives the previous Production version.

This keeps deployment separate from model selection and makes the production rollout decision auditable.

### Monitoring

The production backend exposes Prometheus metrics at `/metrics`.

Required metrics exposed by the backend include:

- total prediction requests
- prediction request latency
- failed prediction requests
- backend uptime
- backend health status

Prometheus scrapes the live production backend, and Grafana reads from Prometheus to visualize:

- request volume over time
- prediction latency over time
- error rate and failed requests
- backend health status

Use the provided monitoring stack in [`monitoring/docker-compose.yml`](monitoring/docker-compose.yml) to run Prometheus and Grafana locally against the production endpoint.

## Reproducibility

The project is designed so training and deployment can be repeated from a clean checkout.

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m flask --app backend.api run --host 0.0.0.0 --port 5000
```

### Backend + Postgres

```powershell
docker compose up --build
```

This starts the backend on port 5000 and a separate Postgres container for application users.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Train a Model

```powershell
python train.py --data data/processed/training_table.csv
```

This produces a new MLflow run, logs the trained model, and registers the version in the `Staging` stage.

### Promote a Model

```powershell
python promote.py --data data/processed/training_table.csv
```

This compares the latest Staging model against the current Production model and promotes only if the candidate wins the evaluation gate.

### Run Tests

```powershell
pytest -m unit -v
pytest -m integration -v
pytest -m e2e -v
```

## Production Monitoring Setup

1. Start the monitoring stack with `docker compose -f monitoring/docker-compose.yml up -d`.
2. Open Grafana at `http://localhost:3000` and sign in with the Grafana admin credentials from the compose file or your overrides.
3. Open the `Production Backend Overview` dashboard to view request volume, latency, error rate, health, uptime, and model load from the live production backend.

Prometheus scrapes the backend's `/metrics` endpoint on the production deployment, and Grafana uses Prometheus as its data source.