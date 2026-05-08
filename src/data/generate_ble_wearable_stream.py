"""
Wearable BLE Drowsiness Data Generator
=====================================

This script generates a realistic, time-coordinated BLE data stream
originating from a simulated wearable device. The simulation includes
MULTIPLE drowsiness episodes occurring during a single continuous session.

The generated data stream explicitly models:
- Physiological realism (heart rate within normal ranges)
- Multiple coordinated drowsiness events (onset → drowsy → recovery)
- Bluetooth Low Energy (BLE) transmission latency with jitter
- Packet loss events
- Time-aligned sensor-side and phone-side timestamps

The output is stored as a single Excel file inside `src/data/`.

Author: Natalia Naranjo Rodríguez
Course: Mobile Networks and IoT
Date: May 2026
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================
rng = np.random.default_rng(seed=42)

# ============================================================
# PATH CONFIGURATION
# ============================================================
DATA_DIR = os.path.join("src", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "wearable_ble_stream.xlsx")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

DURATION_SECONDS = 10 * 60          # 10 minutes
SAMPLING_PERIOD_SEC = 1             # 1 Hz sampling

# ------------------------------------------------------------
# Drowsiness episodes (seconds)
# Each episode: (onset_start, drowsy_start, drowsy_end, recovery_end)
# ------------------------------------------------------------
DROWSINESS_EPISODES = [
    (40, 60, 90, 120),     # ~1 minute
    (160, 180, 210, 240), # ~3 minutes
    (280, 300, 360, 420)  # ~5 minutes
]

# ------------------------------------------------------------
# Heart rate parameters
# ------------------------------------------------------------
ALERT_HR_MEAN = 76
DROWSY_HR_MEAN = 60
HR_STD_ALERT = 4
HR_STD_DROWSY = 2

# ------------------------------------------------------------
# BLE communication model
# ------------------------------------------------------------
BLE_LATENCY_MEAN_MS = 90
BLE_LATENCY_STD_MS = 30
BLE_LATENCY_MIN_MS = 20
PACKET_LOSS_PROBABILITY = 0.03

# ============================================================
# HELPER FUNCTION
# ============================================================

def compute_drowsiness_level(t_sec):
    """
    Compute the latent drowsiness level (0.0 → 1.0) at time t_sec
    considering multiple drowsiness episodes.
    """
    for onset, d_start, d_end, recover_end in DROWSINESS_EPISODES:

        # Onset phase
        if onset <= t_sec < d_start:
            return (t_sec - onset) / (d_start - onset)

        # Fully drowsy phase
        if d_start <= t_sec <= d_end:
            return 1.0

        # Recovery phase
        if d_end < t_sec <= recover_end:
            return max(
                0.2,
                1.0 - (t_sec - d_end) / (recover_end - d_end)
            )

    # Default: alert state
    return 0.1

# ============================================================
# DATA GENERATION
# ============================================================

rows = []

for t_sec in range(0, DURATION_SECONDS, SAMPLING_PERIOD_SEC):
    t_sensor_ms = t_sec * 1000

    # Latent drowsiness state
    drowsiness_level = compute_drowsiness_level(t_sec)

    # --------------------------------------------------------
    # Heart rate generation (coordinated)
    # --------------------------------------------------------
    hr_mean = (
        ALERT_HR_MEAN * (1 - drowsiness_level) +
        DROWSY_HR_MEAN * drowsiness_level
    )

    hr_std = (
        HR_STD_ALERT * (1 - drowsiness_level) +
        HR_STD_DROWSY * drowsiness_level
    )

    heart_rate = rng.normal(hr_mean, hr_std)
    heart_rate = max(40, int(round(heart_rate)))

    # --------------------------------------------------------
    # BLE latency simulation
    # --------------------------------------------------------
    ble_latency_ms = max(
        BLE_LATENCY_MIN_MS,
        rng.normal(BLE_LATENCY_MEAN_MS, BLE_LATENCY_STD_MS)
    )
    ble_latency_ms = round(ble_latency_ms, 1)

    # --------------------------------------------------------
    # Packet loss simulation
    # --------------------------------------------------------
    packet_lost = 1 if rng.random() < PACKET_LOSS_PROBABILITY else 0

    # --------------------------------------------------------
    # Arrival timestamp at smartphone
    # --------------------------------------------------------
    t_arrival_ms = t_sensor_ms + ble_latency_ms

    rows.append({
        "t_sensor_ms": t_sensor_ms,
        "heart_rate_bpm": heart_rate,
        "drowsiness_level": round(drowsiness_level, 2),
        "ble_latency_ms": ble_latency_ms,
        "packet_lost": packet_lost,
        "t_arrival_ms": t_arrival_ms
    })

# ============================================================
# EXPORT TO EXCEL
# ============================================================

df = pd.DataFrame(rows)

df.to_excel(
    OUTPUT_FILE,
    sheet_name="BLE_Stream",
    index=False,
    engine="openpyxl"
)

print("Wearable BLE drowsiness simulation generated successfully")
print(f"Output file: {OUTPUT_FILE}")
print("Drowsiness events at ~1, ~3 and ~5 minutes")
print(f"Total samples: {len(df)}")