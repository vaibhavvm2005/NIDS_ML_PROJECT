#!/usr/bin/env python3
"""
ML-NIDS Live Detection with Arduino Buzzer
- Sniffs network traffic and predicts attacks (Random Forest)
- Provides HTTP endpoint for dashboard alerts (POST /alert)
"""

import sys
import time
import json
import threading
import logging
from collections import defaultdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd
import joblib

# ============================================================================
#  CONFIGURATION – CHANGE THESE TWO LINES FOR YOUR SYSTEM
# ============================================================================
# Windows: find your Arduino port in Device Manager (e.g., "COM3", "COM4")
# Linux:   "/dev/ttyACM0" or "/dev/ttyUSB0"
ARDUINO_PORT = 'COM3'          # <-- CHANGE THIS

# Windows: Leave as None to auto-detect, or set manually after running the
#          helper below. Run: python live_detection.py --list-interfaces
# Linux:   "eth
# 0" or "wlan0"
INTERFACE = None               # <-- SET THIS or use --list-interfaces flag

# Paths to your trained model and label encoder (relative to this script)
MODEL_PATH = '../models/random_forest_nids.pkl'
ENCODER_PATH = '../models/label_encoder.pkl'

# HTTP server port for dashboard alerts
HTTP_PORT = 5001

# Flow management
FLOW_TIMEOUT = 30               # seconds before expiring idle flows
MIN_PACKETS_PER_FLOW = 10       # number of packets to collect before prediction
# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nids-live")

# ----------------------------------------------------------------------------
# FIX Bug 1 & 2: Normalize incoming attack labels from the dashboard.
# Dashboard sends labels like "SSH Brute Force", "Port Scan", etc.
# CMD_MAP keys must match after normalization.
# ----------------------------------------------------------------------------
CMD_MAP = {
    'normal':           b'N',
    'ssh brute force':  b'B',
    'ssh_bruteforce':   b'B',   # legacy key kept for compatibility
    'port scan':        b'S',
    'port_scan':        b'S',
    'reverse shell':    b'R',
    'reverse_shell':    b'R',
    'c2 beacon':        b'C',
    'c2_beacon':        b'C',
    'c2 beaconing':     b'C',
}

def normalize_label(label: str) -> str:
    """Lowercase and strip the label for consistent CMD_MAP lookup."""
    return label.strip().lower()

# ----------------------------------------------------------------------------
# Arduino communication
# ----------------------------------------------------------------------------
arduino = None

def init_arduino():
    global arduino
    try:
        import serial
        arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
        time.sleep(2)          # allow Arduino to reset
        logger.info(f"Arduino connected on {ARDUINO_PORT}")
    except Exception as e:
        logger.warning(f"Arduino not found: {e} – alarms disabled")

def send_arduino_command(attack_label: str):
    """Send a single-byte command to Arduino to trigger the buzzer."""
    if arduino is None:
        return
    # FIX Bug 2: normalize before lookup so "SSH Brute Force" → "ssh brute force"
    key = normalize_label(attack_label)
    cmd = CMD_MAP.get(key)
    if cmd:
        try:
            arduino.write(cmd)
            logger.info(f"Sent '{attack_label}' -> buzzer ({cmd})")
        except Exception as e:
            logger.error(f"Serial error: {e}")
    else:
        logger.warning(f"No buzzer command mapped for label: '{attack_label}' (key='{key}')")

# ----------------------------------------------------------------------------
# FIX Bug 3: Interface auto-detection for Windows (NPcap) and Linux
# ----------------------------------------------------------------------------
def list_interfaces():
    """Print available Scapy interfaces and exit."""
    try:
        from scapy.all import get_if_list, conf
        logger.info("Available interfaces:")
        for iface in get_if_list():
            logger.info(f"  {iface}")
        # On Windows, also show the friendly name mapping
        try:
            from scapy.arch.windows import get_windows_if_list
            logger.info("\nWindows friendly names:")
            for iface in get_windows_if_list():
                logger.info(f"  {iface.get('name')} -> {iface.get('description')}")
        except ImportError:
            pass
    except Exception as e:
        logger.error(f"Could not list interfaces: {e}")
    sys.exit(0)


def resolve_interface(requested: str) -> str:
    """
    On Windows, Scapy needs the NPcap GUID interface name, not the
    friendly name like 'Wi-Fi'. This function tries to resolve it.
    Returns the resolved interface name or the original if no match found.
    """
    if requested is None:
        # Auto-select first non-loopback interface
        try:
            from scapy.all import get_if_list
            ifaces = [i for i in get_if_list() if 'lo' not in i.lower()]
            if ifaces:
                resolved = ifaces[0]
                logger.info(f"Auto-selected interface: {resolved}")
                return resolved
        except Exception:
            pass
        logger.error(
            "Could not auto-detect interface. "
            "Run: python live_detection.py --list-interfaces"
        )
        sys.exit(1)

    # Try to match Windows friendly name → NPcap name
    try:
        from scapy.arch.windows import get_windows_if_list
        for iface in get_windows_if_list():
            friendly = iface.get('name', '')
            desc     = iface.get('description', '')
            if requested.lower() in friendly.lower() or requested.lower() in desc.lower():
                logger.info(f"Resolved '{requested}' -> '{friendly}'")
                return friendly
    except ImportError:
        pass  # Not Windows or no mapping available

    return requested  # Use as-is on Linux


# ----------------------------------------------------------------------------
# Load ML model
# ----------------------------------------------------------------------------
logger.info("Loading ML model...")
try:
    model   = joblib.load(MODEL_PATH)
    le      = joblib.load(ENCODER_PATH)
    logger.info(f"Model loaded. Classes: {list(le.classes_)}")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}. Check MODEL_PATH and ENCODER_PATH.")
    sys.exit(1)

feature_cols = [
    'dst_port', 'protocol', 'total_packets', 'total_bytes',
    'packet_size_mean', 'packet_size_std', 'flow_duration',
    'packets_per_sec', 'bytes_per_sec', 'unique_dst_ports',
    'inter_arrival_mean', 'inter_arrival_std'
]

# ----------------------------------------------------------------------------
# Flow management and feature extraction
# ----------------------------------------------------------------------------
active_flows      = defaultdict(lambda: {'packets': [], 'start_time': time.time()})
active_flows_lock = threading.Lock()   # thread-safe access from HTTP + sniffer threads


def extract_features(packets):
    """Convert list of packet dicts into a feature DataFrame."""
    if len(packets) < 2:
        return None

    times     = [p['time']     for p in packets]
    sizes     = [p['size']     for p in packets]
    ports     = [p['dst_port'] for p in packets]
    protocols = [p['protocol'] for p in packets]

    flow_duration = max(times) - min(times)
    if flow_duration <= 0:
        flow_duration = 1e-6

    total_bytes   = sum(sizes)
    total_packets = len(packets)

    sorted_times  = sorted(times)
    inter_arrival = [
        sorted_times[i + 1] - sorted_times[i]
        for i in range(len(sorted_times) - 1)
    ]

    features = {
        'dst_port':           max(set(ports), key=ports.count),
        'protocol':           1 if protocols[0] == 'TCP' else 0,
        'total_packets':      total_packets,
        'total_bytes':        total_bytes,
        'packet_size_mean':   total_bytes / total_packets,
        'packet_size_std':    pd.Series(sizes).std() if len(sizes) > 1 else 0.0,
        'flow_duration':      flow_duration,
        'packets_per_sec':    total_packets / flow_duration,
        'bytes_per_sec':      total_bytes / flow_duration,
        'unique_dst_ports':   len(set(ports)),
        'inter_arrival_mean': sum(inter_arrival) / len(inter_arrival) if inter_arrival else 0.0,
        'inter_arrival_std':  pd.Series(inter_arrival).std() if len(inter_arrival) > 1 else 0.0,
    }
    return pd.DataFrame([features])[feature_cols]


def process_packet(packet):
    """Called by Scapy for each captured packet."""
    try:
        from scapy.all import IP, TCP, UDP
        if IP not in packet or (TCP not in packet and UDP not in packet):
            return

        timestamp = time.time()
        src_ip    = packet[IP].src
        dst_ip    = packet[IP].dst

        if TCP in packet:
            proto = 'TCP'
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        else:
            proto = 'UDP'
            sport = packet[UDP].sport
            dport = packet[UDP].dport

        flow_key = f"{src_ip}:{sport}->{dst_ip}:{dport}/{proto}"

        with active_flows_lock:
            flow = active_flows[flow_key]
            flow['packets'].append({
                'time':     timestamp,
                'size':     len(packet),
                'dst_port': dport,
                'protocol': proto,
            })

            # When we have enough packets, run prediction
            if len(flow['packets']) >= MIN_PACKETS_PER_FLOW:
                features_df = extract_features(flow['packets'])
                if features_df is not None:
                    # FIX Bug 5: use argmax of proba instead of raw pred_idx
                    # to safely index the probability array.
                    probs      = model.predict_proba(features_df)[0]
                    pred_pos   = int(probs.argmax())          # position in proba array
                    pred_label = le.inverse_transform([pred_pos])[0]
                    confidence = probs[pred_pos] * 100
                    now_str    = datetime.now().strftime("%H:%M:%S")

                    if normalize_label(pred_label) != 'normal':
                        logger.warning(
                            f"\n{'='*60}\n"
                            f"🔴 [{now_str}] ALERT: {pred_label.upper()} DETECTED!\n"
                            f"   Confidence: {confidence:.1f}%\n"
                            f"   {src_ip}:{sport} -> {dst_ip}:{dport}\n"
                            f"{'='*60}"
                        )
                        send_arduino_command(pred_label)
                    else:
                        logger.info(
                            f"🟢 [{now_str}] Normal | "
                            f"{src_ip}:{sport} -> {dst_ip}:{dport} | {confidence:.1f}%"
                        )

                # FIX Bug 4: delete AFTER we finish using the flow data,
                # not mid-iteration. Using pop() avoids KeyError on re-entry.
                active_flows.pop(flow_key, None)

            # Periodically remove stale flows (collect keys first, then delete)
            now     = time.time()
            expired = [
                k for k, v in active_flows.items()
                if now - v['start_time'] > FLOW_TIMEOUT
            ]
            for k in expired:
                active_flows.pop(k, None)   # FIX Bug 4: pop() never raises KeyError

    except Exception as e:
        logger.error(f"Error processing packet: {e}")


# ----------------------------------------------------------------------------
# HTTP server for external alerts (dashboard)
# FIX Bug 1, 2, 6: proper label normalization, Content-Type header, error detail
# ----------------------------------------------------------------------------
class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/alert':
            length = int(self.headers.get('Content-Length', 0))
            body   = self.rfile.read(length)
            try:
                data   = json.loads(body)
                attack = data.get('attack', '')

                # FIX Bug 1 & 2: normalize the incoming label before CMD_MAP lookup
                key = normalize_label(attack)
                cmd = CMD_MAP.get(key)

                if attack and cmd is not None:
                    logger.info(f"External alert received: '{attack}' (key='{key}')")
                    send_arduino_command(attack)
                    self._respond(200, 'OK')
                else:
                    logger.warning(
                        f"Unrecognised attack label from dashboard: '{attack}' "
                        f"(normalized='{key}'). "
                        f"Valid keys: {list(CMD_MAP.keys())}"
                    )
                    self._respond(400, f"Invalid attack label: '{attack}'")

            except json.JSONDecodeError:
                logger.error("Malformed JSON in alert request")
                self._respond(400, 'Malformed JSON')
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                self._respond(500, str(e))
        else:
            self._respond(404, 'Not found')

    def _respond(self, code: int, message: str):
        """FIX Bug 6: always send Content-Type before end_headers()."""
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode())

    def log_message(self, format, *args):
        pass  # suppress verbose HTTP logs


def start_http_server():
    server = HTTPServer(('0.0.0.0', HTTP_PORT), AlertHandler)
    logger.info(f"HTTP server listening on port {HTTP_PORT}")
    server.serve_forever()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # FIX Bug 7: support --list-interfaces flag for easy interface discovery
    if '--list-interfaces' in sys.argv:
        list_interfaces()

    init_arduino()

    # FIX Bug 3: resolve friendly name → NPcap name on Windows
    from scapy.all import sniff
    resolved_iface = resolve_interface(INTERFACE)

    # Start HTTP server in background thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    logger.info("=" * 60)
    logger.info("🛡️  ML-NIDS LIVE DETECTION WITH BUZZER")
    logger.info("=" * 60)
    logger.info(f"Sniffing interface : {resolved_iface}")
    logger.info(f"HTTP endpoint      : http://localhost:{HTTP_PORT}/alert")
    logger.info(f"Min packets/flow   : {MIN_PACKETS_PER_FLOW}")
    logger.info(f"Flow timeout       : {FLOW_TIMEOUT}s")
    logger.info("Press Ctrl+C to stop\n")

    try:
        sniff(iface=resolved_iface, prn=process_packet, store=0)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if arduino:
            arduino.close()
        sys.exit(0)
    except Exception as e:
        logger.error(
            f"Sniff failed on interface '{resolved_iface}': {e}\n"
            f"Try running: python live_detection.py --list-interfaces"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()