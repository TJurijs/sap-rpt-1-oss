# SAP RPT Playground

Interactive playground for the `SAP/sap-rpt-1-oss` tabular in-context learner. The app combines a FastAPI backend with a React/Vite frontend so you can upload datasets, monitor progress live, and download scored test results. The scripts and instructions were tested on macOS; Windows users may need to adapt paths, shell commands, or dependency setup.

## Requirements

- Python 3.11
- Node.js 20 or newer
- Hugging Face account with access to `SAP/sap-rpt-1-oss`
- `HUGGINGFACE_API_KEY` saved locally (see the `.env` section below)
- `curl` and `tar` (needed by `scripts/dev.sh` to fetch the model snapshot)

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
   The script ensures a local copy of the `sap-rpt-1-oss` sources is available (downloading a snapshot if needed), bootstraps a Python virtual environment under `playground/backend/.venv`, installs backend and frontend dependencies (including the model in editable mode), exports `PYTHONPATH`, starts the FastAPI API on port `8000`, waits for it to become healthy, and then launches the Vite dev server on port `5173` (with WebSocket proxying enabled).

4. **Open the UI**  
   Browse to [http://localhost:5173](http://localhost:5173). The status banner confirms Hugging Face connectivity, checkpoint cache state (`sap-rpt-1-oss cached`), and whether the estimator runs on GPU or CPU. A comprehensive progress indicator shows stage, percentage, and ETA while jobs run.

5. **Stop the playground**  
   Press `Ctrl+C` in the same terminal. The script cleans up the background backend process automatically.

## sap-rpt-1-oss Snapshot

- On first launch, `scripts/dev.sh` downloads and unpacks the `sap-rpt-1-oss` repository into a plain directory (`./sap-rpt-1-oss/`) so it can be installed in editable mode without creating a nested Git repo. Subsequent runs reuse the folder.
- The directory is listed in `.gitignore`; remove it manually if you need to refresh or switch branches.
- Override the download source by setting `SAP_RPT_SOURCE_URL` (for example, to point at a specific commit tarball) before running the script.
- The backend uses the local snapshot when instantiating estimators; no additional installation is required after the script finishes.

## Working with Datasets

- Upload CSV, Parquet, or JSON files directly from the Dataset panel. Sample datasets live under `example_datasets/`, but you still pick the files manually in the UI.
- The preview auto-selects the last column as the initial target. If the filename contains `classification` or `regression`, the task type pre-fills accordingly (you can still change it).
- When a run finishes, download the scored test split from the Results dashboard. Filenames follow `<dataset_name> - results.csv`; regression predictions are formatted with a comma decimal separator to simplify spreadsheet viewing.
- Tune task type, target column, test split, maximum context size, bagging factor, and preprocessing options in the Configuration panel before starting a job.

### Included Example Datasets

Each CSV in `example_datasets/` demonstrates a different SAP RPT scenario you can load manually through the Dataset panel:

- **Predictive business outcomes**
  - `predictive_business_outcomes_iInvoice Late_classification.csv`: Will this invoice be paid late? → `late_payment_flag` (classification)
  - `predictive_business_outcomes_days_to_payment_regression.csv`: How many days until payment? → `days_to_pay` (regression)
- **Recommendations & auto-defaulting**
  - `recommendations_form_of_address_classification.csv`: Recommend `form_of_address` for PA0002 (classification)
- **Normalization & coding**
  - `normalization_raw_country_country_iso_code_classification.csv`: Normalize `raw_country` → `country_iso_code` PA0006 (classification)
- **Data quality & anomaly flags**
  - `data_quality_bank_details_needs_review_classification.csv`: Classify `needs_review` (0/1) for bank details PA0009 (classification)
- **Derived scores, segments & priorities**
  - `derived_scores_risk_of_leave_regression.csv`: Set risk score `risk_of_leave` for an employee (regression)
- **Matching & linking via pair-rows**
  - `matching_materials_is_same_entity_classification.csv`: Two materials → `is_same_entity` (0/1) (classification)
- **Information extraction from text**
  - `information_extraction_ticket_topic_classification.csv`: From `ticket_text` → `topic` (classification)

## Backend & Services

- FastAPI exposes REST endpoints for health checks, dataset preview/upload, and job execution, plus WebSockets for real-time progress streaming.
- Job orchestration dispatches inference tasks asynchronously, surfaces granular progress updates, and persists results for later download.
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

