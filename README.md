# 🛡️ ML-NIDS Pro — Machine Learning Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time network intrusion detection system powered by a **Random Forest classifier** that detects SSH Brute Force, Port Scan, Reverse Shell, and C2 Beaconing attacks from live network traffic — with an Arduino buzzer alarm and a professional Streamlit dashboard.

---

## 📸 Dashboard Preview

| Page | Description |
|------|-------------|
| 📊 Overview | Live traffic feed, donut chart, 24h timeline, recent threats |
| 🌍 Threat Map | Global attack origin map with city rankings |
| 📈 Model Performance | Confusion matrix, ROC-AUC, radar chart, training history |
| 🎯 Live Simulator | Attack scenario inference with SHAP attribution |
| 📋 Forensics | Flow inspector, anomaly heatmap, IAT violin plots |

---

## 📁 Project Structure

```
nids_ml_project/
│
├── datasets/
│   ├── c2_beaconing.pcap           # C2 beacon attack captures
│   ├── normal_traffic.pcap         # Benign traffic captures
│   ├── port_scan.pcap              # Port scan attack captures
│   ├── reverse_shell.pcap          # Reverse shell attack captures
│   ├── ssh_bruteforce.pcap         # SSH brute force captures
│   └── network_traffic_dataset.csv # Extracted feature dataset (321 records)
│
├── models/
│   ├── random_forest_nids.pkl      # Trained Random Forest classifier
│   └── label_encoder.pkl           # Class name ↔ index encoder
│
└── scripts/
    ├── logs/                        # Detection logs directory
    ├── venv/                        # Python virtual environment
    ├── extract_features.py          # Step 1 — .pcap → feature CSV
    ├── train_model.py               # Step 2 — Train & save RF model
    ├── live_detection.py            # Step 3 — Live sniffer + HTTP server
    └── dashboard.py                 # Step 4 — Streamlit dashboard
```

---

## ⚙️ Features

- ✅ **Real-time packet sniffing** using Scapy on any network interface
- ✅ **Random Forest classifier** — 100% accuracy on test set (97 samples)
- ✅ **5-class detection** — Normal, SSH Brute Force, Port Scan, Reverse Shell, C2 Beacon
- ✅ **Arduino buzzer alarm** — physical alert on attack detection
- ✅ **Professional Streamlit dashboard** with 5 pages
- ✅ **SHAP-style feature attribution** per inference
- ✅ **HTTP REST endpoint** (`POST /alert`) for external integrations
- ✅ **MITRE ATT&CK mapping** in Forensics page
- ✅ **Auto-block mode** — configurable via dashboard sidebar

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/nids_ml_project.git
cd nids_ml_project/scripts

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install streamlit plotly pandas numpy scikit-learn joblib scapy pyserial requests
```

### 3. Extract Features from PCAPs

```bash
python extract_features.py
```
> Converts all `.pcap` files in `/datasets/` into `network_traffic_dataset.csv`

### 4. Train the Model

```bash
python train_model.py
```
> Saves `random_forest_nids.pkl` and `label_encoder.pkl` into `/models/`

### 5. Find Your Network Interface (Windows)

```bash
python live_detection.py --list-interfaces
```
> Copy the interface name (e.g., `\Device\NPF_{XXXX...}`) and paste it into `live_detection.py`

### 6. Configure `live_detection.py`

Open `live_detection.py` and edit these two lines:

```python
ARDUINO_PORT = 'COM3'       # Your Arduino COM port (check Device Manager)
INTERFACE    = 'Wi-Fi'      # Your network interface name
```

### 7. Run — Open 2 Terminals

**Terminal 1** — Live Detection *(run as Administrator on Windows)*
```bash
cd nids_ml_project/scripts
venv\Scripts\activate
python live_detection.py
```

**Terminal 2** — Dashboard
```bash
cd nids_ml_project/scripts
venv\Scripts\activate
streamlit run dashboard.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Dataset size | 321 records |
| Train / Test split | 70% / 30% |
| Test accuracy | **100%** |
| Inference speed | ~1.8 ms average |
| Features used | 14 network flow features |

### Detected Attack Classes

| Class | Description | MITRE ATT&CK |
|-------|-------------|--------------|
| `normal` | Benign network traffic | — |
| `ssh_bruteforce` | Repeated SSH login attempts | T1110.001 |
| `port_scan` | Systematic port enumeration | T1046 |
| `reverse_shell` | Outbound shell connection | T1059 |
| `c2_beaconing` | Periodic C2 communication | T1071 |

### Top 5 Important Features

| Feature | Importance |
|---------|-----------|
| `packet_size_std` | 0.233 |
| `total_bytes` | 0.222 |
| `packet_size_mean` | 0.178 |
| `dst_port` | 0.095 |
| `flow_duration` | 0.069 |

---

## 🔌 Arduino Setup

The system sends single-byte commands to an Arduino to trigger a buzzer on attack detection.

| Attack | Byte sent |
|--------|-----------|
| Normal | `N` |
| SSH Brute Force | `B` |
| Port Scan | `S` |
| Reverse Shell | `R` |
| C2 Beacon | `C` |

**Wiring:** Connect a buzzer to digital pin 8 and GND on your Arduino. Upload a sketch that reads the serial byte and activates the buzzer accordingly.

---

## 🌐 HTTP API

`live_detection.py` exposes a REST endpoint for the dashboard to trigger Arduino alarms:

```
POST http://localhost:5001/alert
Content-Type: application/json

{ "attack": "SSH Brute Force" }
```

**Response:** `200 OK` on success, `400` for invalid label, `500` on server error.

**Test it:**
```bash
curl -X POST http://localhost:5001/alert \
  -H "Content-Type: application/json" \
  -d "{\"attack\": \"SSH Brute Force\"}"
```

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Arduino not found: COM6` | Check Device Manager → Ports for correct COM port |
| `Address already in use: 5001` | `netstat -ano \| findstr 5001` then `taskkill /PID <id> /F` |
| Scapy captures nothing | Run terminal as **Administrator**; use `--list-interfaces` to verify interface name |
| `Model file not found` | Run `train_model.py` first |
| Dashboard shows warning | Start `live_detection.py` before the dashboard |
| 100% accuracy (overfitting) | Dataset is small (321 records) — collect more pcap data for production use |

---

## 📦 Requirements

```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
scapy>=2.5.0
pyserial>=3.5
requests>=2.31.0
```

Save as `requirements.txt` and install with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Development

### Run Order (Always follow this sequence)

```
extract_features.py  →  train_model.py  →  live_detection.py  →  dashboard.py
      (Step 1)              (Step 2)            (Step 3)             (Step 4)
```

### Adding a New Attack Class

1. Add new `.pcap` captures to `/datasets/`
2. Label them in `extract_features.py`
3. Retrain with `train_model.py`
4. Add the new class to `CMD_MAP` in `live_detection.py`
5. Update `CLASSES` and `CLASS_COLORS` in `dashboard.py`

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vaibhav**
- Project: ML-NIDS Pro
- Stack: Python · Scapy · Scikit-Learn · Streamlit · Arduino

---

> ⚠️ **Disclaimer:** This tool is intended for educational and authorized network monitoring purposes only. Always obtain proper authorization before monitoring any network.