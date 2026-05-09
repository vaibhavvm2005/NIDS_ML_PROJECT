#!/usr/bin/env python3
"""
FEATURE EXTRACTION SCRIPT
Converts .pcap files → labeled CSV dataset
"""

from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import os
from collections import defaultdict


def extract_features(pcap_file, label):
    """Extract 13 features from each network flow in a pcap file"""

    packets = rdpcap(pcap_file)
    flows = defaultdict(list)

    # Group packets into flows (same src_ip:port → dst_ip:port)
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            if TCP in pkt:
                proto  = 'TCP'
                sport  = pkt[TCP].sport
                dport  = pkt[TCP].dport
                flags  = pkt[TCP].flags
            else:
                proto  = 'UDP'
                sport  = pkt[UDP].sport
                dport  = pkt[UDP].dport
                flags  = 0

            flow_key = f"{pkt[IP].src}:{sport}->{pkt[IP].dst}:{dport}/{proto}"
            flows[flow_key].append({
                'time'    : float(pkt.time),
                'size'    : len(pkt),
                'flags'   : str(flags),
                'src_port': sport,
                'dst_port': dport,
                'protocol': proto
            })

    # Extract flow-level features
    features = []
    for flow_key, pkts in flows.items():
        if len(pkts) < 2:          # Skip single-packet flows
            continue

        times = [p['time']     for p in pkts]
        sizes = [p['size']     for p in pkts]
        ports = [p['dst_port'] for p in pkts]

        flow_duration = max(times) - min(times)
        total_packets = len(pkts)
        total_bytes   = sum(sizes)

        # Inter-arrival times
        times_sorted  = sorted(times)
        inter_arrival = [times_sorted[i+1] - times_sorted[i]
                         for i in range(len(times_sorted) - 1)]

        # SYN / RST flag counts (TCP only)
        syn_count = sum(1 for p in pkts if 'S' in str(p['flags']))
        rst_count = sum(1 for p in pkts if 'R' in str(p['flags']))

        features.append({
            'src_port'          : pkts[0]['src_port'],
            'dst_port'          : max(set(ports), key=ports.count),
            'protocol'          : 1 if pkts[0]['protocol'] == 'TCP' else 0,
            'total_packets'     : total_packets,
            'total_bytes'       : total_bytes,
            'packet_size_mean'  : sum(sizes) / len(sizes) if sizes else 0,
            'packet_size_std'   : pd.Series(sizes).std() if len(sizes) > 1 else 0,
            'flow_duration'     : flow_duration,
            'packets_per_sec'   : total_packets / flow_duration if flow_duration > 0 else 0,
            'bytes_per_sec'     : total_bytes   / flow_duration if flow_duration > 0 else 0,
            'unique_dst_ports'  : len(set(ports)),
            'inter_arrival_mean': sum(inter_arrival) / len(inter_arrival) if inter_arrival else 0,
            'inter_arrival_std' : pd.Series(inter_arrival).std() if len(inter_arrival) > 1 else 0,
            'syn_count'         : syn_count,
            'rst_count'         : rst_count,
            'label'             : label
        })

    return features


# ─────────────────────────────────────────────
# MAIN — EXTRACT FROM ALL PCAP FILES
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("🔄 Extracting features...\n")
    all_features = []

    # Path to datasets folder (one level up from scripts/)
    datasets_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

    pcap_configs = [
        ('normal_traffic.pcap' , 'normal'       ),
        ('ssh_bruteforce.pcap' , 'ssh_bruteforce'),
        ('port_scan.pcap'      , 'port_scan'     ),
        ('reverse_shell.pcap'  , 'reverse_shell' ),
        ('c2_beaconing.pcap'   , 'c2_beaconing'  ),   # ← fixed filename
    ]

    for filename, label in pcap_configs:
        pcap_path = os.path.join(datasets_dir, filename)
        if os.path.exists(pcap_path):
            print(f"  📦 Processing {filename} → label: '{label}'")
            flows = extract_features(pcap_path, label)
            all_features.extend(flows)
            print(f"     ✅ Extracted {len(flows)} flows\n")
        else:
            print(f"  ⚠️  File not found: {pcap_path}\n")

    # ── Save CSV ──────────────────────────────
    if all_features:
        df       = pd.DataFrame(all_features)
        csv_path = os.path.join(datasets_dir, 'network_traffic_dataset.csv')
        df.to_csv(csv_path, index=False)

        print("=" * 50)
        print(f"✅ DATASET CREATED — {len(df)} total flow records")
        print(f"\n📊 Class distribution:")
        print(df['label'].value_counts().to_string())
        print(f"\n📋 Features  : {[c for c in df.columns if c != 'label']}")
        print(f"📁 Saved to  : {csv_path}")
        print("=" * 50)
    else:
        print("❌ No features extracted. Check that pcap files are in the datasets/ folder.")