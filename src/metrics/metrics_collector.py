import os
import time
import numpy as np
import pandas as pd
from enum import Enum
from collections import deque


# ============================================================
# ENUM FOR FUSED STATE (TO AVOID CIRCULAR IMPORTS)
# ============================================================

class FusedDriverState(Enum):
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPY = "SLEEPY"


# ============================================================
# METRICS COLLECTOR CLASS
# ============================================================

class MetricsCollector:
    """
    Collector for evaluation metrics in drowsiness detection system.
    Handles quality metrics (delays, coincidences), resource metrics (CPU),
    and temporal evolution logging.

    This class is designed for academic projects: explicit, well-commented,
    and justifiable. All decisions are based on physiological realism and
    safety-first principles.
    """

    def __init__(self, max_history=1000):
        """
        Initialize the metrics collector.

        Args:
            max_history (int): Maximum number of data points to keep in memory
                              for rolling averages and temporal analysis.
        """
        # --------------------------------------------------------
        # Temporal evolution storage
        # --------------------------------------------------------
        self.timestamps = deque(maxlen=max_history)
        self.visual_states = deque(maxlen=max_history)
        self.wearable_drowsy_flags = deque(maxlen=max_history)
        self.fused_states = deque(maxlen=max_history)
        self.cpu_usages = deque(maxlen=max_history)
        self.energy_estimates = deque(maxlen=max_history)  # Cumulative energy estimates

        # --------------------------------------------------------
        # Detection delay tracking
        # --------------------------------------------------------
        self.wearable_drowsy_timestamps = []  # When wearable first detects DROWSY
        self.visual_drowsy_timestamps = []    # When visual detects DROWSY
        self.fused_drowsy_timestamps = []     # When fused detects DROWSY

        # --------------------------------------------------------
        # Coincidence counters (for percentage calculations)
        # --------------------------------------------------------
        self.total_frames = 0
        self.coincidence_count = 0  # Frames where visual and wearable agree on DROWSY/AWAKE

        # --------------------------------------------------------
        # Resource metrics accumulators
        # --------------------------------------------------------
        self.cpu_sum = 0.0
        self.energy_estimate = 0.0  # Simple estimate in CPU-seconds

        # --------------------------------------------------------
        # Session start time for relative timestamps
        # --------------------------------------------------------
        self.session_start = time.time()

    def update(self, current_time, visual_state, wearable_drowsy, fused_state, cpu_percent):
        """
        Update metrics with new frame data.

        Args:
            current_time (float): Current session time in seconds.
            visual_state (DriverState): Current visual model state.
            wearable_drowsy (bool): True if wearable indicates DROWSY.
            fused_state (FusedDriverState): Current fused model state.
            cpu_percent (float): Current CPU usage percentage.
        """
        # Store temporal data
        self.timestamps.append(current_time)
        self.visual_states.append(visual_state)
        self.wearable_drowsy_flags.append(wearable_drowsy)
        self.fused_states.append(fused_state)
        self.cpu_usages.append(cpu_percent)

        # Update coincidence counter
        self.total_frames += 1
        visual_drowsy = visual_state == visual_state.DROWSY or visual_state == visual_state.SLEEPY
        if visual_drowsy == wearable_drowsy:
            self.coincidence_count += 1

        # Track first detections for delay calculation
        if wearable_drowsy and not self.wearable_drowsy_timestamps:
            self.wearable_drowsy_timestamps.append(current_time)
        if visual_drowsy and not self.visual_drowsy_timestamps:
            self.visual_drowsy_timestamps.append(current_time)
        if fused_state == FusedDriverState.DROWSY and not self.fused_drowsy_timestamps:
            self.fused_drowsy_timestamps.append(current_time)

        # Accumulate resource metrics
        self.cpu_sum += cpu_percent
        self.energy_estimate += cpu_percent / 100.0  # Simple estimate: 1% CPU = 0.01 energy units
        self.energy_estimates.append(self.energy_estimate)

    def get_average_cpu(self):
        """
        Calculate average CPU usage percentage.

        Returns:
            float: Average CPU % over collected frames.
        """
        if not self.cpu_usages:
            return 0.0
        return self.cpu_sum / len(self.cpu_usages)

    def get_energy_estimate(self):
        """
        Get simple energy consumption estimate in CPU-seconds.

        Returns:
            float: Total energy estimate.
        """
        return self.energy_estimate

    def get_detection_delay(self, model='fused', current_time=None):
        """
        Calculate detection delay: time from wearable DROWSY to model detection.

        Args:
            model (str): 'visual' or 'fused'.
            current_time (float, optional): Current session time in seconds.

        Returns:
            float: Delay in seconds, or None if wearable never detected DROWSY.
        """
        if not self.wearable_drowsy_timestamps:
            return None

        wearable_time = self.wearable_drowsy_timestamps[0]
        if model == 'visual' and self.visual_drowsy_timestamps:
            return self.visual_drowsy_timestamps[0] - wearable_time
        elif model == 'fused' and self.fused_drowsy_timestamps:
            return self.fused_drowsy_timestamps[0] - wearable_time

        if current_time is not None:
            return current_time - wearable_time
        return None

    def get_coincidence_percentage(self):
        """
        Calculate percentage of frames where visual and wearable agree.

        Returns:
            float: Coincidence percentage (0-100).
        """
        if self.total_frames == 0:
            return 0.0
        return (self.coincidence_count / self.total_frames) * 100.0

    def get_temporal_events(self, last_n=10):
        """
        Get recent temporal events for timeline display.

        Args:
            last_n (int): Number of recent events to return.

        Returns:
            list: List of strings describing recent state changes.
        """
        events = []
        recent = list(zip(self.timestamps, self.visual_states, self.wearable_drowsy_flags, self.fused_states))[-last_n:]
        for t, vis, wear, fus in recent:
            wearable_text = 'DROWSY' if wear else 'AWAKE'
            events.append(
                f"{t:.1f}s: Visual={vis.value}, Wearable={wearable_text}, Fused={fus.value}"
            )
        return events

    def export_to_csv(self, filepath):
        """
        Export collected data to CSV for post-analysis.

        Args:
            filepath (str): Path to save CSV file.
        """
        data = {
            'timestamp': list(self.timestamps),
            'visual_state': [s.value for s in self.visual_states],
            'wearable_drowsy': list(self.wearable_drowsy_flags),
            'fused_state': [s.value for s in self.fused_states],
            'cpu_percent': list(self.cpu_usages),
            'energy_estimate': list(self.energy_estimates)
        }
        df = pd.DataFrame(data)
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        df.to_csv(filepath, index=False)