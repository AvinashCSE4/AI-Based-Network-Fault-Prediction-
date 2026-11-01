# NetGuard AI — AI-Based Network Fault Prediction

Computer Networks mini project. A Streamlit dashboard that uses a Random Forest model (compared with
Decision Tree and Logistic Regression) to classify network conditions as **Normal / Warning / Critical**.

> **Honest scope:** the model is trained on **synthetic data** whose labels come from a documented rule-based
> function. The dashboard shows **simulated measurements** (Demo Mode) or **stored CSV records**; it does not
> connect to real routers/switches or Cisco Packet Tracer. A "Critical" result is a *prediction*, not a confirmed failure.
> No e-mail/SMS notifications are implemented — alerts appear inside the dashboard.

## Project structure
```
AI_Network_Fault_Prediction/
├── app.py              # Streamlit UI (7 pages)
├── train_model.py      # data generation, labelling, training, prediction, CSV validation
├── requirements.txt
├── README.md
├── .streamlit/config.toml   # light blue theme
├── data/network_dataset.csv # sample dataset (created automatically)
├── models/fault_prediction_model.pkl  # saved with Joblib (created automatically)
└── assets/style.css, logo.svg
```

## Windows installation (VS Code)
1. Install **Python 3.10 – 3.12** from python.org (tick *"Add Python to PATH"*) and **VS Code** with the *Python* extension.
2. In VS Code: **File → Open Folder…** → choose `AI_Network_Fault_Prediction`.
3. Open the terminal: **Terminal → New Terminal** (PowerShell).
4. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
5. In VS Code press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick the `venv` interpreter.
6. Install the libraries:
   ```powershell
   pip install -r requirements.txt
   ```
7. (Optional) train the model manually — the app also does this automatically on first run:
   ```powershell
   python train_model.py
   ```
8. **Run the dashboard:**
   ```powershell
   streamlit run app.py
   ```
   The browser opens at http://localhost:8501.

## Demo script for your professor
1. **Overview** – KPI cards, health gauge, active alerts (all from simulated data; the header says so).
2. **Network Monitoring** – topology and device cards with predicted status. Change *Demo network condition* in the sidebar.
3. **Simulation Lab** – click *Load: Normal / Warning / Critical Network*, then **Run AI Fault Prediction**. Move a slider and
   watch the "sliders changed" notice. **Reset Simulation** restores defaults.
4. **Fault History** – filter, download CSV. **Network Analytics** – 7 Plotly charts.
5. **AI Fault Prediction → Model Performance** – accuracy, precision, recall, F1, confusion matrix, algorithm comparison.
6. **CSV Upload** (sidebar) – upload `data/network_dataset.csv` or your own file.

## CSV format
Required columns: `latency, packet_loss, bandwidth_utilization, throughput, cpu_utilization, memory_utilization, network_traffic, error_rate`.
Optional: `timestamp, device, device_type, connection_status` and a label column (`fault_label`, `label`, `target` or `condition`
with values Normal/Warning/Critical). If valid labels exist (≥ 50 rows, ≥ 2 classes, ≥ 5 per class) a button lets you retrain the
model on them; otherwise the saved model is used. Missing columns, empty files, non-numeric values, missing values and out-of-range
values produce clear error messages.

## Changing the assumptions
Thresholds, value ranges and the scoring rule are in `FEATURE_INFO`, `WARNING_MIN_POINTS` and `CRITICAL_MIN_POINTS` at the top of
`train_model.py`. Delete `models/fault_prediction_model.pkl` (or run `python train_model.py`) after editing so the model is retrained.

## Troubleshooting
* `streamlit` not recognised → activate the venv (step 4) and rerun `pip install -r requirements.txt`.
* Port busy → `streamlit run app.py --server.port 8502`.
* Fonts missing offline → the app falls back to Segoe UI / Consolas; everything still works.
