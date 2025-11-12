# SAP RPT Playground

Interactive playground for the `SAP/sap-rpt-1-oss` tabular in-context learner. The app combines a FastAPI backend with a React/Vite frontend so you can upload datasets, monitor progress live, and download scored test results.

## Requirements

- Python 3.11
- Node.js 20 or newer
- Hugging Face account with access to `SAP/sap-rpt-1-oss`
- `HUGGINGFACE_API_KEY` saved locally (see the `.env` section below)

## Quickstart

1. **Clone the repository**
   ```bash
   git clone https://github.com/Krimik/sap-rpt-1-oss.git
   cd sap-rpt-1-oss
   ```

2. **Configure environment variables**
   ```bash
   cp env.example .env
   # edit .env and paste your HUGGINGFACE_API_KEY
   chmod +x scripts/dev.sh
   ```

3. **Launch the playground**
   ```bash
   ./scripts/dev.sh
   ```
   The script bootstraps a Python virtual environment under `playground/backend/.venv`, installs backend and frontend dependencies, exports `PYTHONPATH`, starts the FastAPI API on port `8000`, waits for it to become healthy, and then launches the Vite dev server on port `5173` (with WebSocket proxying enabled).

4. **Open the UI**  
   Browse to [http://localhost:5173](http://localhost:5173). The status banner confirms Hugging Face connectivity, checkpoint cache state (`sap-rpt-1-oss cached`), and whether the estimator runs on GPU or CPU. A comprehensive progress indicator shows stage, percentage, and ETA while jobs run.

5. **Stop the playground**  
   Press `Ctrl+C` in the same terminal. The script cleans up the background backend process automatically.

## Working with Datasets

- Upload CSV, Parquet, or JSON files directly from the Dataset panel, or place files in the `example_datasets/` directory and they will be listed automatically.
- When a run finishes, download the scored test split from the Results dashboard. Filenames follow the pattern `<dataset_name> - results.csv`.
- Adjust task type, target column, test split, maximum context size, bagging factor, and preprocessing options in the Configuration panel before starting a job.

## Backend & Services

- FastAPI exposes REST endpoints for health checks, dataset upload, example dataset management, and job execution, plus WebSockets for real-time progress streaming.
- Job orchestration dispatches inference tasks asynchronously and persists results for later download.
- Hugging Face authentication is required to download the checkpoint once; subsequent runs reuse the cached copy.
- The ZeroMQ embedding server is launched and managed automatically when the estimator starts.

## Troubleshooting

- **Model download blocked**: Ensure you accepted the model license on Hugging Face and confirm `HUGGINGFACE_API_KEY` is present in `.env`.
- **OOM on CPU**: Lower `max_context_size` or `bagging` in the Configuration panel. Default values (`1024` and `2`) are tuned for CPU-friendly runs.
- **WebSocket errors**: Make sure you start both backend and frontend using `./scripts/dev.sh`; it configures the Vite proxy (`ws: true`) for streaming updates.

## Development Notes

- No Docker configuration is included; the project targets local development only.
- The repo is structured under `playground/backend` and `playground/frontend`. Edit React components or FastAPI routes there as needed.
- Keep personal filesystem paths out of documentation and configuration. Use project-relative paths (e.g., `./example_datasets`) when sharing setup steps.

Happy experimenting! Send feedback or open issues on GitHub if you run into problems.
