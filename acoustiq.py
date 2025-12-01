#!/usr/bin/env python3

"""
Fit N peaking IIR filters to the correction curve using nonlinear least squares.
Normalizes both measurement and target so that ref_hz = 0 dB, optionally anchors
the filter stack to 0 dB at ref_hz. Generates a Plotly plot (or interactive PyQt6
with --interactive) showing:
- Original measurement
- Target
- EQ curve applied (what the filters do)
- Corrected measurement (measurement + EQ)
- Error error curve (raw measurement deviation from target)
- Shaded residuals

For zero filters (-p 0 without --high-shelf), outputs a minimal result and plots
only the measurement and target curves for review.

This version caches the complex z = exp(-j*2*pi*f/fs) vectors so the expensive
exponential is computed once per frequency grid instead of per-biquad per-iteration.

Optionally writes the corrected measurement to a file with 'frequency,raw' headers.
Optionally reads initial filter parameters from a file with --filter-input.

Filter labels (e.g., PK 1, HS 8) act as buttons to toggle filters on/off, with a green LED indicator when on.
Filter parameters (Fc, Q, Gain) are editable via QLineEdit below each QDial, with Enter key triggering updates.
Static QLabels (Fc, Q, Gain) are added above QLineEdit fields for clarity, center-aligned.
A Reoptimize button reruns the optimization using current GUI parameters as initial guesses, only accepting results with lower RMSE.
Filter parameters are output with precision matching GUI step sizes (Fc: 0.01 Hz, Q: 0.001, Gain: 0.01 dB).
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import math
import plotly.graph_objects as go
import sys
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                            QDial, QLabel, QPushButton, QLineEdit)
from PyQt6.QtCore import Qt, QRectF, QPointF  # ADDED: QPointF
from PyQt6.QtGui import (QDoubleValidator, QPainter, QPainterPath, QConicalGradient,
                        QRadialGradient, QPen, QBrush, QColor, QFont)
import pyqtgraph as pg

# =======================
# Optimizer Settings
# =======================
OPT_SETTINGS = {
    "fs": 48000.0,
    "n_peaks": 9,
    "maxiter": 100000,
    "freq_grid_points": 2048,
    "bounds": {
        "fc": (20.0, 20000.0),
        "Q": (0.1, 10.0),
        "shelf_Q": (0.01, 1.0),
        "gain_db": (-12.0, 12.0)
    }
}

class GlowDial(QDial):
    """Professional analog mixer-style knob with dynamic glowing arc for EQ controls."""

    def __init__(self, parent=None, glow_color="#00ff88", knob_color="#2a2a2a",
                 text_color="#ffffff", accent_color="#00ff88", value_font_size=9):
        super().__init__(parent)
        self.glow_color = QColor(glow_color)
        self.knob_color = QColor(knob_color)
        self.text_color = QColor(text_color)
        self.accent_color = QColor(accent_color)
        self.value_font_size = value_font_size

        self.setNotchesVisible(False)
        self.setMinimumSize(70, 70)
        self.setWrapping(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._shadow_color = QColor(0, 0, 0, 200)
        self._highlight_color = QColor(255, 255, 255, 40)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:  # ADDED: Proper cleanup
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            size = min(self.width(), self.height())
            center = self.rect().center()
            outer_radius = size / 2 - 4
            knob_radius = outer_radius * 0.65

            rect = QRectF(center.x() - outer_radius, center.y() - outer_radius,
                        outer_radius * 2, outer_radius * 2)
            knob_rect = QRectF(center.x() - knob_radius, center.y() - knob_radius,
                            knob_radius * 2, knob_radius * 2)

            # Calculate knob position (270° sweep, starting at 135°)
            span_angle = 270
            start_angle = 135
            angle_range = self.maximum() - self.minimum()
            if angle_range > 0:
                angle = (self.value() - self.minimum()) / angle_range * span_angle
            else:
                angle = 0

            # === 1. BACKGROUND RING ===
            bg_path = QPainterPath()
            bg_path.addEllipse(rect)
            bg_grad = QConicalGradient(QPointF(center), -45)
            bg_grad.setColorAt(0.0, QColor(50, 50, 50))
            bg_grad.setColorAt(0.3, self.knob_color.lighter(110))
            bg_grad.setColorAt(0.7, self.knob_color.darker(120))
            bg_grad.setColorAt(1.0, QColor(30, 30, 30))
            painter.fillPath(bg_path, QBrush(bg_grad))

            painter.setPen(QPen(QColor("#e0b060"), 1.5))
            painter.drawEllipse(rect.adjusted(-1, -1, 1, 1))

            # === 2. GLOWING ARC ===
            arc_rect = rect.adjusted(6, 6, -6, -6)
            glow_pen = QPen(self.glow_color, 6)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            start_arc = int((90 - (start_angle + angle)) * 16)
            span_arc = int(angle * 16)
            painter.drawArc(arc_rect, start_arc, span_arc)

            # === 3. OUTER GLOW ===
            painter.setPen(QPen(self.glow_color.lighter(150), 12))
            painter.setOpacity(0.3)
            painter.drawArc(arc_rect.adjusted(-2, -2, 2, 2), start_arc, span_arc)
            painter.setOpacity(1.0)

            # === 4. TICK MARKS ===
            painter.setPen(QPen(self.accent_color, 2))
            tick_rect = rect.adjusted(8, 8, -8, -8)

            for i in range(5):
                tick_angle = start_angle + (i * span_angle / 4) - 90
                rad = math.radians(tick_angle)
                x1 = center.x() + tick_rect.width() / 2 * math.cos(rad)
                y1 = center.y() + tick_rect.height() / 2 * math.sin(rad)
                x2 = center.x() + (tick_rect.width() / 2 - 8) * math.cos(rad)
                y2 = center.y() + (tick_rect.height() / 2 - 8) * math.sin(rad)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # === 5. MAIN KNOB ===
            knob_grad = QRadialGradient(center.x(), center.y(), knob_radius)
            knob_grad.setColorAt(0.0, QColor("#e0e0b0"))  # Lightened #e0b060 for center
            knob_grad.setColorAt(0.4, QColor("#d0a050"))  # Mid-tone blend
            knob_grad.setColorAt(0.8, self.knob_color.darker(140))  # Darker edge
            knob_grad.setColorAt(1.0, self.knob_color.darker(180))  # Darkest edge
            painter.setBrush(QBrush(knob_grad))
            painter.setPen(QPen(self._shadow_color, 2))
            painter.drawEllipse(knob_rect)

            painter.setPen(QPen(QColor("#e0b060"), 1.5))
            painter.drawArc(knob_rect.adjusted(1, 1, -2, -2), 0, 360 * 16)

            # === 6. KNOB INDICATOR ===
            painter.setPen(QPen(QColor("#e0b060"), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            indicator_length = int(knob_radius * 0.55)  # FIXED: Cast to int

            painter.save()
            painter.translate(center)
            painter.rotate(start_angle + angle)
            painter.drawLine(0, 0, 0, -indicator_length)  # NOW ALL INTS
            painter.restore()

            # === 7. CENTER DOT ===
            center_dot_grad = QRadialGradient(QPointF(center), 4)
            center_dot_grad.setColorAt(0.0, self.accent_color)
            center_dot_grad.setColorAt(0.6, self.accent_color.darker(120))
            center_dot_grad.setColorAt(1.0, self.accent_color.darker(160))
            painter.setBrush(QBrush(center_dot_grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center.x() - 3, center.y() - 3, 6, 6)

            # === 8. VALUE LABEL ===
            #painter.setPen(QPen(self.text_color, 2))
            #font = painter.font()
            #font.setPixelSize(self.value_font_size)
            #painter.setFont(font)
            #
            #value_str = f"{self.value():.0f}"
            #painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, value_str)

        finally:
            painter.end()  # GUARANTEED CLEANUP

# ---------------------------
# Peaking Biquad (RBJ)
# ---------------------------
def peaking_biquad(fc, Q, gain_db, fs):
    A = 10**(gain_db / 40.0)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2.0 * Q)
    cosw0 = math.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

# ---------------------------
# High-shelf biquad (RBJ)
# ---------------------------
def high_shelf_biquad(fc, Q, gain_db, fs):
    A = 10**(gain_db / 40.0)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2.0 * Q)
    cosw0 = math.cos(w0)
    sqrtA = math.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * sqrtA * alpha)
    b1 = -2*A * ((A - 1) + (A + 1) * cosw0)
    b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * sqrtA * alpha)
    a0 = (A + 1) - (A - 1) * cosw0 + 2 * sqrtA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw0)
    a2 = (A + 1) - (A - 1) * cosw0 - 2 * sqrtA * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

# ---------------------------
# Low-shelf biquad (RBJ)
# ---------------------------
def low_shelf_biquad(fc, Q, gain_db, fs):
    """RBJ low-shelf (gain > 0 dB boosts low frequencies)."""
    A  = 10**(gain_db / 40.0)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2.0 * Q)
    cosw0 = math.cos(w0)
    sqrtA = math.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cosw0 + 2 * sqrtA * alpha)
    b1 = 2*A * ((A - 1) - (A + 1) * cosw0)
    b2 = A * ((A + 1) - (A - 1) * cosw0 - 2 * sqrtA * alpha)
    a0 = (A + 1) + (A - 1) * cosw0 + 2 * sqrtA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cosw0)
    a2 = (A + 1) + (A - 1) * cosw0 - 2 * sqrtA * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

# NOTE: Accept precomputed z vector instead of recomputing exp(-j*w) each time
def biquad_freq_response(b, a, z):
    z1 = z**-1
    z2 = z**-2
    num = b[0] + b[1] * z1 + b[2] * z2
    den = 1.0 + a[1] * z1 + a[2] * z2
    return num / den

def total_response(params, z, fs, n_filters, use_high_shelf=False, use_low_shelf=False, filter_states=None):
    H = np.ones_like(z, dtype=np.complex128)
    for i in range(n_filters):
        if filter_states is not None and not filter_states[i]:
            continue  # Skip disabled filters
        fc, Q, g = params[i*3:(i+1)*3]
        if use_high_shelf and i == (n_filters - 1):
            b, a = high_shelf_biquad(fc, Q, g, fs)
        elif use_low_shelf and i == 0:
            b, a = low_shelf_biquad(fc, Q, g, fs)
        else:
            b, a = peaking_biquad(fc, Q, g, fs)
        H *= biquad_freq_response(b, a, z)
    return H

def residuals(param_vec, z_fit, desired_linear, n_filters, weight,
              anchor_enable=True, anchor_weight=50.0, ref_hz=1000.0,
              use_high_shelf=False, use_low_shelf=False, filter_states=None, fs=48000.0):
    """Calculate residuals in energy domain (magnitude squared)"""

    # Build full params (disabled filters keep initial values)
    full_params = param_vec.copy()
    if filter_states is not None:
        j = 0
        for i in range(n_filters):
            if filter_states[i]:
                full_params[i*3:(i+1)*3] = param_vec[j*3:(j+1)*3]
                j += 1

    # Compute full response
    H = total_response(full_params, z_fit, fs, n_filters, use_high_shelf, use_low_shelf, filter_states)
    mag_linear = np.abs(H)

    # Residuals in energy domain
    mag_energy = mag_linear**2
    desired_energy = desired_linear**2
    res = (mag_energy - desired_energy) * weight

    # Anchor constraint at reference frequency
    if anchor_enable:
        z_ref = np.exp(-1j * 2.0 * math.pi * ref_hz / fs)
        h_ref = total_response(full_params, np.array([z_ref]), fs, n_filters, use_high_shelf, use_low_shelf, filter_states)[0]
        mag_ref_linear = np.abs(h_ref)
        mag_ref_energy = mag_ref_linear**2
        res = np.concatenate([res, np.array([(mag_ref_energy - 1.0) * anchor_weight])])

    return res, [], []

def pretty_filters_from_params(params, n_filters, filter_states, filter_types):
    filters = []
    for i in range(n_filters):
        fc, Q, g = params[i*3:(i+1)*3]
        typ = filter_types[i]
        state = 'ON' if filter_states[i] else 'OFF'
        filters.append((typ, fc, Q, g, state))
    return filters

def smooth_vector(x, kernel_size):
    if kernel_size <= 1:
        return x
    kernel = np.ones(kernel_size) / kernel_size
    pad = kernel_size // 2
    xp = np.pad(x, pad, mode='edge')
    sm = np.convolve(xp, kernel, mode='valid')
    return sm

def parse_bounds(s):
    try:
        vals = [float(x.strip()) for x in s.split(",")]
        if len(vals) != 2:
            raise ValueError(f"Invalid bounds format: '{s}'. Expected 'low,high'.")
        if vals[0] >= vals[1]:
            raise ValueError(f"Invalid bounds: '{s}'. Lower bound must be less than upper bound.")
        return tuple(vals)
    except ValueError as e:
        print(f"Error parsing bounds '{s}': {str(e)}")
        sys.exit(1)

def parse_filter_file(filename, expected_n_filters, use_high_shelf, use_low_shelf):
    """
    Parse a filter file (e.g., MMX300-Neutral-7-96.txt) and extract parameters.
    Expected format per line: Filter N: ON|OFF {PK|HS} Fc X Hz Gain Y dB Q Z
    Returns: p0 (initial parameters), lb (lower bounds), ub (upper bounds), filter_states
    """
    filters = []
    filter_regex = re.compile(r"Filter \d+: (ON|OFF) (PK|LS|HS) Fc ([\d.]+) Hz Gain ([-+]?[\d.]+) dB Q ([\d.]+)")

    try:
        with open(filename, 'r') as f:
            for line in f:
                match = filter_regex.match(line.strip())
                if match:
                    state, typ, fc, gain, q = match.groups()
                    filters.append((typ, float(fc), float(q), float(gain), state))
    except FileNotFoundError:
        error_exit(f"Filter input file not found: {filename}")
    except Exception as e:
        error_exit(f"Failed to read filter input file {filename}: {str(e)}")

    if len(filters) != expected_n_filters:
        print(f"Warning: Filter file {filename} contains {len(filters)} filters, but expected {expected_n_filters}. Adjusting parameters.")
        # Pad with default filters if too few
        while len(filters) < expected_n_filters:
            last_fc = filters[-1][1] * 1.1 if filters else 100.0
            last_fc = min(last_fc, OPT_SETTINGS["bounds"]["fc"][1])
            filters.append(('PK', last_fc, 0.707, 0.0, 'ON'))
        # Truncate if too many
        filters = filters[:expected_n_filters]

    if use_low_shelf and filters[0][0] != 'LS':
        error_exit(f"First filter in {filename} must be LS when --low-shelf is used, got {filters[0][0]}.")
    if use_high_shelf and filters[-1][0] != 'HS':
        error_exit(f"Last filter in {filename} must be HS when --high-shelf is used, got {filters[-1][0]}.")
    if not use_high_shelf and any(f[0] == 'HS' for f in filters):
        error_exit(f"Filter file {filename} contains HS filter, but --high-shelf is not specified.")

    filters.sort(key=lambda x: x[1])

    p0_list = []
    lb_list = []
    ub_list = []
    filter_states = []

    for i, (typ, fc, q, gain, state) in enumerate(filters):
        p0_list.extend([fc, q, gain])
        filter_states.append(state == 'ON')
        if typ == 'HS':
            lb_list.extend([fc * 0.5, OPT_SETTINGS["bounds"]["shelf_Q"][0], OPT_SETTINGS["bounds"]["gain_db"][0]])
            ub_list.extend([min(fc * 2.0, OPT_SETTINGS["fs"] / 2.0), OPT_SETTINGS["bounds"]["shelf_Q"][1], OPT_SETTINGS["bounds"]["gain_db"][1]])
        else:
            lb_list.extend([OPT_SETTINGS["bounds"]["fc"][0], OPT_SETTINGS["bounds"]["Q"][0], OPT_SETTINGS["bounds"]["gain_db"][0]])
            ub_list.extend([OPT_SETTINGS["bounds"]["fc"][1], OPT_SETTINGS["bounds"]["Q"][1], OPT_SETTINGS["bounds"]["gain_db"][1]])

    return np.array(p0_list), np.array(lb_list), np.array(ub_list), filter_states

def error_exit(msg):
    print(f"Error: {msg}")
    sys.exit(1)

class EQWindow(QMainWindow):
    def __init__(self, params, lb, ub, n_filters, freqs_full, z_full, meas_abs, target_abs,
                use_high_shelf, use_low_shelf, fs, corrected_output,
                desired_linear_fit, desired_db_fit, filter_states=None, z_fit=None,
                weight_fit=None, anchor_enable=True, anchor_weight=50.0, ref_hz=1000.0,
                source_file=None, target_file=None,
                n_peaks=None,
                fc_bounds_peaks_str=None,
                Q_bounds_str=None,
                low_shelf_fc=None,
                high_shelf_fc=None,
                fc_bounds_low_shelf_str=None,
                fc_bounds_shelf_str=None,
                output_file=None,
                fc_peaks_low=None,
                fc_peaks_high=None,
                fc_low_shelf_low=None,
                fc_low_shelf_high=None,
                fc_shelf_low=None,
                fc_shelf_high=None):
        super().__init__()

        self.use_low_shelf = use_low_shelf
        self.use_high_shelf = use_high_shelf
        self.low_shelf_fc = low_shelf_fc
        self.high_shelf_fc = high_shelf_fc

        # Store desired_linear_fit
        self.desired_linear_fit = desired_linear_fit

        # -----------------------------------------------------------------
        # SORT ALL FILTERS BY CENTER FREQUENCY (Fc)
        # -----------------------------------------------------------------
        filter_data = []
        for i in range(n_filters):
            fc = params[i*3]
            is_low_shelf = use_low_shelf and i == 0
            is_high_shelf = use_high_shelf and i == (n_filters - 1)
            typ = 'LS' if is_low_shelf else 'HS' if is_high_shelf else 'PK'
            filter_data.append((
                fc,
                params[i*3:(i+1)*3],
                lb[i*3:(i+1)*3],
                ub[i*3:(i+1)*3],
                filter_states[i] if filter_states is not None else True,
                i,
                typ  # Add type
            ))

        # Sort purely by Fc (first element)
        filter_data.sort(key=lambda x: x[0])

        # Rebuild sorted arrays
        self.params = np.concatenate([f[1] for f in filter_data])
        self.lb = np.concatenate([f[2] for f in filter_data])
        self.ub = np.concatenate([f[3] for f in filter_data])
        self.filter_states = [f[4] for f in filter_data]
        self.filter_types = [f[6] for f in filter_data]

        # Update index mapping
        orig_idx_list = [f[5] for f in filter_data]
        self.original_to_sorted = {orig: new for new, orig in enumerate(orig_idx_list)}
        self.sorted_to_original = {new: orig for new, orig in enumerate(orig_idx_list)}

        self.n_filters = n_filters
        self.freqs_full = freqs_full
        self.z_full = z_full
        self.meas_abs = meas_abs
        self.target_abs = target_abs
        self.fs = fs
        self.corrected_output = corrected_output
        self.desired_db_fit = desired_db_fit
        self.z_fit = z_fit
        self.weight_fit = weight_fit
        self.anchor_enable = anchor_enable
        self.anchor_weight = anchor_weight
        self.ref_hz = ref_hz
        self.source_file = source_file
        self.target_file = target_file
        self.is_optimizing = False
        self.best_rmse = float('inf')
        self.best_params = self.params.copy()
        self.best_filter_states = self.filter_states.copy()
        self.desired_linear_fit = desired_linear_fit
        self.desired_db_fit = desired_db_fit
        self.output_file = output_file  # Store output file path
        # === Store CLI values for re-optimization ===
        self.n_peaks = n_peaks
        self.fc_bounds_peaks_str = fc_bounds_peaks_str
        self.Q_bounds_str = Q_bounds_str
        self.low_shelf_fc = low_shelf_fc
        self.high_shelf_fc = high_shelf_fc
        self.fc_bounds_low_shelf_str = fc_bounds_low_shelf_str
        self.fc_bounds_shelf_str = fc_bounds_shelf_str
        # Store filter frequency bounds
        self.fc_peaks_low = fc_peaks_low
        self.fc_peaks_high = fc_peaks_high
        self.fc_low_shelf_low = fc_low_shelf_low
        self.fc_low_shelf_high = fc_low_shelf_high
        self.fc_shelf_low = fc_shelf_low
        self.fc_shelf_high = fc_shelf_high

        self.setWindowTitle("AcoustiQ")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            background-color: #2b2b2b; color: #e0b060;
            QLabel { font-size: 11px; margin: 1px; }
            QDial { width: 70px; height: 70px; margin: 2px; }
            QPushButton { font-weight: bold; padding: 6px; background-color: #b88a2a; color: #1c1c1c; border-radius: 3px; max-width: 150px; }
            QPushButton:hover { background-color: #d1a23a; }
            QPushButton:disabled { background-color: #666; color: #999; }
            QLineEdit { font-size: 11px; padding: 2px; width: 75px; background-color: #3a3a3a; border: 1px solid #7a5f1a; border-radius: 3px; }
            QLineEdit:focus { border: 1px solid #e0b060; }
        """)

        # -----------------------------------------------------------------
        # REBUILD fit_mask USING GLOBAL BOUNDS (after sorting and freqs_full is set)
        # -----------------------------------------------------------------
        mask_peaks = (self.freqs_full >= self.fc_peaks_low) & (self.freqs_full <= self.fc_peaks_high)
        mask_low_shelf = np.zeros_like(self.freqs_full, dtype=bool)
        mask_high_shelf = np.zeros_like(self.freqs_full, dtype=bool)

        if self.use_low_shelf and self.fc_low_shelf_low is not None:
            mask_low_shelf = (self.freqs_full >= self.fc_low_shelf_low) & (self.freqs_full <= self.fc_low_shelf_high)
        if self.use_high_shelf and self.fc_shelf_low is not None:
            mask_high_shelf = (self.freqs_full >= self.fc_shelf_low) & (self.freqs_full <= self.fc_shelf_high)

        self.fit_mask = mask_peaks | mask_low_shelf | mask_high_shelf

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Plot setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1f1f1f')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.addLegend(offset=(-10, 10))
        gold = (224, 176, 96)
        dark_gold = (122, 95, 26)
        axis_pen = pg.mkPen(color=gold, width=1.2)
        tick_font = QFont("Arial", 11)
        for axis in ['left', 'bottom']:
            ax = self.plot_widget.getAxis(axis)
            ax.setPen(axis_pen)
            ax.setTextPen(pg.mkPen(color=gold))
            ax.setTickFont(tick_font)
            ax.setStyle(tickTextOffset=10)
        self.plot_widget.setLabel('left', 'Amplitude (dB)', color='#e0b060')
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)', color='#e0b060')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        vb = self.plot_widget.getPlotItem().getViewBox()
        vb.setBorder(pg.mkPen(gold))
        for item in self.plot_widget.getPlotItem().listDataItems():
            item.setPen(pg.mkPen(color=dark_gold, width=0.8))
        main_layout.addWidget(self.plot_widget)

        # Initial plot
        self.meas_plot = self.plot_widget.plot(self.freqs_full, self.meas_abs, pen=pg.mkPen('deepskyblue', width=2), name='Measurement', antialias=True)
        self.tgt_plot = self.plot_widget.plot(self.freqs_full, self.target_abs, pen=pg.mkPen('gold', width=2), name='Target', antialias=True)
        H = total_response(self.params, self.z_full, self.fs, self.n_filters, self.use_high_shelf, self.use_low_shelf, self.filter_states)
        mag_linear = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(mag_linear, 1e-12))
        self.best_rmse_linear = np.sqrt(np.mean((mag_linear[self.fit_mask] - self.desired_linear_fit)**2))
        self.best_rmse_db = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
        corrected_db = self.meas_abs + mag_db
        error_db = corrected_db - self.target_abs
        self.eq_plot = self.plot_widget.plot(self.freqs_full, mag_db, pen=pg.mkPen('dodgerblue', width=2), name='EQ Curve', antialias=True)
        self.corr_plot = self.plot_widget.plot(self.freqs_full, corrected_db, pen=pg.mkPen('lime', width=2, style=Qt.PenStyle.DotLine), name='Corrected', antialias=True)
        self.err_plot = self.plot_widget.plot(self.freqs_full, error_db, pen=pg.mkPen('orange', width=2, style=Qt.PenStyle.DashLine), name='Error', antialias=True)

        self.best_rmse = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        self.best_params = self.params.copy()
        self.best_filter_states = self.filter_states.copy()
        self.rmse_label = QLabel(f"RMSE: {self.best_rmse_linear:.4f} — {self.best_rmse_db:.4f} dB")
        self.rmse_label.setStyleSheet("font-family: 'Menlo'; font-size: 14px; background-color: #1f1f1f; color: #ffd580; padding: 8px; border-radius: 4px; border: 1px solid #e0b060;")
        self.rmse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.rmse_label, stretch=1)

        self.reoptimize_button = QPushButton("Reoptimize")
        self.reoptimize_button.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e0b060; color: #1f1f1f; max-width: 150px;")
        self.reoptimize_button.clicked.connect(self.reoptimize)
        control_layout.addWidget(self.reoptimize_button, stretch=0)
        main_layout.addLayout(control_layout)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        self.dials = []
        self.filter_buttons = []
        self.leds = []
        self.line_edits = []
        self.gain_labels = []
        for i in range(n_filters):
            v_layout = QVBoxLayout()
            v_layout.setSpacing(4)
            typ = self.filter_types[i]  # Use preserved type

            # Filter toggle button and LED
            h_layout = QHBoxLayout()
            h_layout.setSpacing(5)
            filter_button = QPushButton(f"{typ} {i+1}")
            filter_button.setStyleSheet("font-weight: bold; padding: 4px 8px; min-width: 60px;")
            filter_button.clicked.connect(lambda checked, idx=i: self.update_filter_state(idx))
            filter_button.setEnabled(True)
            h_layout.addWidget(filter_button)
            self.filter_buttons.append(filter_button)

            led = QLabel()
            led.setFixedSize(12, 12)
            led.setStyleSheet(f"background-color: {'#44ff44' if self.filter_states[i] else '#ff4444'}; border-radius: 6px; border: 1px solid #333;")
            h_layout.addWidget(led)
            self.leds.append(led)
            h_layout.addStretch()
            v_layout.addLayout(h_layout)

            # Fc control
            fc_label = QLabel("Fc (Hz)")
            fc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fc_label.setStyleSheet("font-weight: bold; color: #33bbff;")
            v_layout.addWidget(fc_label)

            fc_edit = QLineEdit(f"{self.params[3*i]:.2f}")
            fc_validator = QDoubleValidator(self.lb[3*i], self.ub[3*i], 2)
            fc_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            fc_edit.setValidator(fc_validator)
            fc_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fc_edit.setStyleSheet("background-color: #3a3a3a; border: 1px solid #7a5f1a; border-radius: 3px; color: #33bbff;")
            fc_edit.editingFinished.connect(lambda idx=i, edit=fc_edit: self.update_from_line_edit(idx * 3, edit, "{:.2f}", 100))
            fc_edit.returnPressed.connect(lambda idx=i, edit=fc_edit: self.update_from_line_edit(idx * 3, edit, "{:.2f}", 100))
            self.line_edits.append(fc_edit)

            fc_dial = GlowDial(self, glow_color="#00aaff", accent_color="#00aaff")
            fc_dial.setMinimum(int(self.lb[3*i] * 100))
            fc_dial.setMaximum(int(self.ub[3*i] * 100))
            fc_dial.setValue(int(self.params[3*i] * 100))
            fc_dial.valueChanged.connect(lambda v, idx=i, edit=fc_edit: self.update_param(idx * 3, v / 100.0, edit, "{:.2f}", 100))
            v_layout.addWidget(fc_dial)
            self.dials.append(fc_dial)
            v_layout.addWidget(fc_edit)

            # Q control
            q_label = QLabel("Q")
            q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_label.setStyleSheet("font-weight: bold; color: #d36cff;")
            v_layout.addWidget(q_label)

            q_edit = QLineEdit(f"{self.params[3*i+1]:.3f}")
            q_validator = QDoubleValidator(self.lb[3*i+1], self.ub[3*i+1], 3)
            q_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            q_edit.setValidator(q_validator)
            q_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_edit.setStyleSheet("background-color: #3a3a3a; border: 1px solid #7a5f1a; border-radius: 3px; color: #d36cff;")
            q_edit.editingFinished.connect(lambda idx=i, edit=q_edit: self.update_from_line_edit(idx * 3 + 1, edit, "{:.3f}", 1000))
            q_edit.returnPressed.connect(lambda idx=i, edit=q_edit: self.update_from_line_edit(idx * 3 + 1, edit, "{:.3f}", 1000))
            self.line_edits.append(q_edit)

            q_dial = GlowDial(self, glow_color="#ff44ff", accent_color="#ff44ff")
            q_dial.setMinimum(int(self.lb[3*i+1] * 1000))
            q_dial.setMaximum(int(self.ub[3*i+1] * 1000))
            q_dial.setValue(int(self.params[3*i+1] * 1000))
            q_dial.valueChanged.connect(lambda v, idx=i, edit=q_edit: self.update_param(idx * 3 + 1, v / 1000.0, edit, "{:.3f}", 1000))
            v_layout.addWidget(q_dial)
            self.dials.append(q_dial)
            v_layout.addWidget(q_edit)

            # Gain control
            gain_label = QLabel("Gain (dB)")
            gain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            base_gain = self.params[3*i+2]
            gain_label.setStyleSheet("font-weight: bold; color: " + ("#ff6666" if base_gain < 0 else "#44ff88") + ";")
            v_layout.addWidget(gain_label)
            self.gain_labels.append(gain_label)

            g_edit = QLineEdit(f"{base_gain:.2f}")
            g_validator = QDoubleValidator(self.lb[3*i+2], self.ub[3*i+2], 2)
            g_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            g_edit.setValidator(g_validator)
            g_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gain_color = "#ff6666" if base_gain < 0 else "#44ff88"
            g_edit.setStyleSheet(f"background-color: #3a3a3a; border: 1px solid #7a5f1a; border-radius: 3px; color: {gain_color};")
            g_edit.editingFinished.connect(lambda idx=i, edit=g_edit: self.update_from_line_edit(idx * 3 + 2, edit, "{:.2f}", 100))
            g_edit.returnPressed.connect(lambda idx=i, edit=g_edit: self.update_from_line_edit(idx * 3 + 2, edit, "{:.2f}", 100))
            self.line_edits.append(g_edit)

            gain_glow = "#ff4444" if base_gain < 0 else "#44ff44"
            g_dial = GlowDial(self, glow_color=gain_glow, accent_color=gain_glow)
            g_dial.setMinimum(int(self.lb[3*i+2] * 100))
            g_dial.setMaximum(int(self.ub[3*i+2] * 100))
            g_dial.setValue(int(self.params[3*i+2] * 100))
            g_dial.valueChanged.connect(lambda v, idx=i, edit=g_edit: self.update_param(idx * 3 + 2, v / 100.0, edit, "{:.2f}", 100))
            v_layout.addWidget(g_dial)
            self.dials.append(g_dial)
            v_layout.addWidget(g_edit)

            controls_layout.addLayout(v_layout)
        main_layout.addLayout(controls_layout)

    def update_param(self, idx, value, line_edit, format_str, scale):
        if self.is_optimizing:
            print(f"Ignoring param {idx} update to {value} (optimization in progress)")
            line_edit.setText(format_str.format(self.params[idx]))
            self.dials[idx].setValue(int(self.params[idx] * scale))
            return
        orig_idx = self.sorted_to_original[idx // 3] * 3 + (idx % 3)
        print(f"Updating param {orig_idx} to {value} (dial triggered)")
        self.params[idx] = value

        if idx % 3 == 2:
            new_glow = "#ff4444" if value < 0 else "#44ff44"
            dial = self.dials[idx]
            dial.glow_color = QColor(new_glow)
            dial.accent_color = QColor(new_glow)
            dial.update()
            gain_index = idx // 3
            gain_label = self.gain_labels[gain_index]
            label_color = "#ff6666" if value < 0 else "#44ff88"
            gain_label.setStyleSheet(f"font-weight: bold; color: {label_color};")
            line_edit.setStyleSheet(f"background-color: #3a3a3a; border: 1px solid #7a5f1a; border-radius: 3px; color: {label_color};")

        line_edit.setText(format_str.format(value))
        self.dials[idx].setValue(int(value * scale))
        H = total_response(self.params, self.z_full, self.fs, self.n_filters, self.use_high_shelf, self.use_low_shelf, self.filter_states)
        mag_linear = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(mag_linear, 1e-12))
        rmse_linear = np.sqrt(np.mean((mag_linear[self.fit_mask] - self.desired_linear_fit)**2))
        rmse_db = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
        corrected_db = self.meas_abs + mag_db
        error_db = corrected_db - self.target_abs
        self.eq_plot.setData(self.freqs_full, mag_db)
        self.corr_plot.setData(self.freqs_full, corrected_db)
        self.err_plot.setData(self.freqs_full, error_db)
        #rmse = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        if rmse_linear < self.best_rmse_linear - 1e-6:
            print(f"New best RMSE from param update: {rmse_linear:.4f} — {rmse_db:.4f} dB")
            self.best_rmse_linear = rmse_linear
            self.best_rmse_db = rmse_db
            self.best_params = self.params.copy()
            self.best_filter_states = self.filter_states.copy()
        self.rmse_label.setText(f"RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")
        QApplication.processEvents()

    def update_from_line_edit(self, idx, line_edit, format_str, scale):
        if self.is_optimizing:
            print(f"Ignoring line edit update for param {idx} (optimization in progress)")
            line_edit.setText(format_str.format(self.params[idx]))
            return
        orig_idx = self.sorted_to_original[idx // 3] * 3 + (idx % 3)
        print(f"Processing QLineEdit input for param {orig_idx}: '{line_edit.text()}'")
        try:
            value = float(line_edit.text())
            print(f"Parsed value: {value}, Bounds: [{self.lb[idx]}, {self.ub[idx]}]")
            if self.lb[idx] <= value <= self.ub[idx]:
                print(f"Valid input, updating param {orig_idx} to {value}")
                self.params[idx] = value
                self.dials[idx].setValue(int(value * scale))
                line_edit.setText(format_str.format(value))
                H = total_response(self.params, self.z_full, self.fs, self.n_filters, self.use_high_shelf, self.use_low_shelf, self.filter_states)
                mag_linear = np.abs(H)
                mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
                corrected_db = self.meas_abs + mag_db
                error_db = corrected_db - self.target_abs
                self.eq_plot.setData(self.freqs_full, mag_db)
                self.corr_plot.setData(self.freqs_full, corrected_db)
                self.err_plot.setData(self.freqs_full, error_db)
                rmse_linear = np.sqrt(np.mean((mag_linear[self.fit_mask] - self.desired_linear_fit)**2))
                rmse_db = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
                if rmse_linear < self.best_rmse_linear - 1e-6:
                    print(f"New best RMSE from line edit: {rmse_linear:.4f} — {rmse_db:.4f} dB")
                    self.best_rmse_linear = rmse_linear
                    self.best_rmse_db      = rmse_db
                    self.best_params       = self.params.copy()
                    self.best_filter_states = self.filter_states.copy()

                # ---- GUI label & final console output ----
                self.rmse_label.setText(f"RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")
                print(f"Updated RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")

                QApplication.processEvents()

            else:
                print(f"Input {value} out of bounds, reverting to {self.params[idx]}")
                line_edit.setText(format_str.format(self.params[idx]))
        except ValueError as e:
            print(f"Invalid input '{line_edit.text()}': {str(e)}, reverting to {self.params[idx]}")
            line_edit.setText(format_str.format(self.params[idx]))

    def update_filter_state(self, idx):
        if self.is_optimizing:
            print(f"Ignoring filter {idx+1} toggle (optimization in progress)")
            return
        orig_idx = self.sorted_to_original[idx]
        print(f"Toggling filter {orig_idx+1}, new state: {not self.filter_states[idx]}")
        self.filter_states[idx] = not self.filter_states[idx]
        self.leds[idx].setStyleSheet(f"background-color: {'#44ff44' if self.filter_states[idx] else '#ff4444'}; border-radius: 6px; border: 1px solid #333;")
        H = total_response(self.params, self.z_full, self.fs, self.n_filters, self.use_high_shelf, self.use_low_shelf, self.filter_states)
        mag_linear = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
        corrected_db = self.meas_abs + mag_db
        error_db = corrected_db - self.target_abs
        self.eq_plot.setData(self.freqs_full, mag_db)
        self.corr_plot.setData(self.freqs_full, corrected_db)
        self.err_plot.setData(self.freqs_full, error_db)
        rmse_linear = np.sqrt(np.mean((mag_linear[self.fit_mask] - self.desired_linear_fit)**2))
        rmse_db     = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        if rmse_linear < self.best_rmse_linear - 1e-6:
            print(f"New best RMSE from filter toggle: {rmse_linear:.4f} — {rmse_db:.4f} dB")
            self.best_rmse_linear = rmse_linear
            self.best_rmse_db      = rmse_db
            self.best_params       = self.params.copy()
            self.best_filter_states = self.filter_states.copy()
        self.rmse_label.setText(f"RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")
        QApplication.processEvents()
        print(f"Filter {orig_idx+1} state updated, RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")

    def reoptimize(self):
        current_filter_states = self.filter_states.copy()
        if self.is_optimizing:
            print("Optimization already in progress. Please wait.")
            return
        self.is_optimizing = True
        self.reoptimize_button.setEnabled(False)
        self.reoptimize_button.setText("Optimizing...")
        for button in self.filter_buttons:
            button.setEnabled(False)
        print("Reoptimizing with current parameters as initial guess")

        # === 1. Reuse correct bounds (this is the RIGHT way) ===
        lb_full = self.lb.copy()
        ub_full = self.ub.copy()

        # === 2. Build active params from GUI dials ===
        active_params = []
        active_lb = []
        active_ub = []
        active_indices = []
        x_scale = []

        for i in range(self.n_filters):
            if not current_filter_states[i]:
                continue

            fc = self.dials[i*3].value() / 100.0
            q  = self.dials[i*3+1].value() / 1000.0
            g  = self.dials[i*3+2].value() / 100.0

            active_params.extend([fc, q, g])
            active_lb.extend(lb_full[i*3:(i+1)*3])
            active_ub.extend(ub_full[i*3:(i+1)*3])
            active_indices.extend([i*3, i*3+1, i*3+2])
            x_scale.extend([0.01, 0.001, 0.01])

        if not active_params:
            print("No active filters to optimize.")
            self.is_optimizing = False
            self.reoptimize_button.setEnabled(True)
            self.reoptimize_button.setText("Reoptimize")
            for button in self.filter_buttons:
                button.setEnabled(True)
            return

        active_params = np.array(active_params)
        active_lb = np.array(active_lb)
        active_ub = np.array(active_ub)
        x_scale = np.array(x_scale)

        # Jitter to avoid flat start
        active_params = np.clip(
            active_params + np.random.uniform(-1e-6, 1e-6, size=active_params.shape),
            active_lb + 1e-12, active_ub - 1e-12
        )

        # === 3. Run optimizer ===
        res = None
        try:
            res = least_squares(
                lambda x: residuals(
                    x, self.z_fit, self.desired_linear_fit, self.n_filters,
                    self.weight_fit, self.anchor_enable, self.anchor_weight,
                    self.ref_hz, self.use_high_shelf, self.use_low_shelf,
                    current_filter_states, self.fs
                )[0],
                active_params,
                bounds=(active_lb, active_ub),
                x_scale=x_scale,
                max_nfev=OPT_SETTINGS["maxiter"],
                verbose=2,
                xtol=1e-9, ftol=1e-9, gtol=1e-9,
                jac='3-point', loss='soft_l1', method='trf'
            )

            # Write back results to original order
            j = 0
            for i in range(self.n_filters):
                if current_filter_states[i]:
                    self.params[i*3:(i+1)*3] = res.x[j*3:(j+1)*3]
                    j += 1

        except Exception as e:
            print(f"Optimization failed: {e}")
        finally:
            self.is_optimizing = False
            self.reoptimize_button.setEnabled(True)
            self.reoptimize_button.setText("Reoptimize")
            for button in self.filter_buttons:
                button.setEnabled(True)

        if res is None or not res.success:
            print(f"Warning: Reoptimization did not converge: {getattr(res, 'message', 'Unknown error')}")
            self.rmse_label.setText(f"RMSE: {self.best_rmse_linear:.4f} — {self.best_rmse_db:.4f} dB")
            return

        # === 4. Re-sort ALL filters by Fc ===
        filter_data = []
        for i in range(self.n_filters):
            orig_idx = self.sorted_to_original[i]  # Map back to original index
            is_low_shelf = self.use_low_shelf and orig_idx == 0
            is_high_shelf = self.use_high_shelf and orig_idx == (self.n_filters - 1)
            typ = 'LS' if is_low_shelf else 'HS' if is_high_shelf else 'PK'
            filter_data.append((
                self.params[i*3],
                self.params[i*3:(i+1)*3],
                self.lb[i*3:(i+1)*3],
                self.ub[i*3:(i+1)*3],
                current_filter_states[i],
                i,
                typ
            ))
        filter_data.sort(key=lambda x: x[0])  # Sort by Fc

        new_params = np.concatenate([f[1] for f in filter_data])
        new_states = [f[4] for f in filter_data]
        new_orig_idx = [f[5] for f in filter_data]
        new_orig_to_sorted = {orig: new for new, orig in enumerate(new_orig_idx)}

        # === 5. Evaluate new RMSE ===
        H = total_response(new_params, self.z_full, self.fs, self.n_filters,
                           self.use_high_shelf, self.use_low_shelf, new_states)
        mag_linear = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(mag_linear, 1e-12))
        rmse_linear = np.sqrt(np.mean((mag_linear[self.fit_mask] - self.desired_linear_fit)**2))
        rmse_db = np.sqrt(np.mean((mag_db[self.fit_mask] - self.desired_db_fit)**2))
        print(f"RMSE after optimization: {rmse_linear:.4f} — {rmse_db:.4f} dB")

        # === 6. Accept if better ===
        if rmse_linear < self.best_rmse_linear - 1e-6:
            print(f"Reoptimization improved RMSE: {self.best_rmse_linear:.4f} → {rmse_linear:.4f}")

            # Update state
            self.params = new_params
            self.filter_states = new_states
            self.original_to_sorted = new_orig_to_sorted
            self.sorted_to_original = {v: k for k, v in new_orig_to_sorted.items()}
            self.best_rmse_linear = rmse_linear
            self.best_rmse_db = rmse_db
            self.best_params = new_params.copy()
            self.best_filter_states = new_states.copy()

            # === FULL GUI UPDATE (success) ===
            for i in range(self.n_filters):
                fc, Q, g = self.params[i*3:(i+1)*3]
                state = self.filter_states[i]

                self.line_edits[i*3].setText(f"{fc:.2f}")
                self.line_edits[i*3+1].setText(f"{Q:.3f}")
                self.line_edits[i*3+2].setText(f"{g:.2f}")

                self.dials[i*3].setValue(int(fc * 100))
                self.dials[i*3+1].setValue(int(Q * 1000))
                self.dials[i*3+2].setValue(int(g * 100))

                # LED
                self.leds[i].setStyleSheet(
                    f"background-color: {'#44ff44' if state else '#ff4444'}; "
                    "border-radius: 6px; border: 1px solid #333;"
                )

                # Gain label & text color
                color = "#ff6666" if g < 0 else "#44ff88"
                self.gain_labels[i].setStyleSheet(f"font-weight: bold; color: {color};")
                self.line_edits[i*3+2].setStyleSheet(
                    f"background-color: #3a3a3a; border: 1px solid #7a5f1a; "
                    f"border-radius: 3px; color: {color};"
                )

                # Dial glow
                glow = "#ff4444" if g < 0 else "#44ff44"
                self.dials[i*3+2].glow_color = QColor(glow)
                self.dials[i*3+2].accent_color = QColor(glow)
                self.dials[i*3+2].update()

            # Update plots
            corrected_db = self.meas_abs + mag_db
            error_db = corrected_db - self.target_abs
            self.eq_plot.setData(self.freqs_full, mag_db)
            self.corr_plot.setData(self.freqs_full, corrected_db)
            self.err_plot.setData(self.freqs_full, error_db)
            self.rmse_label.setText(f"RMSE: {rmse_linear:.4f} — {rmse_db:.4f} dB")
            QApplication.processEvents()
            print("GUI updated with new best parameters.")

        else:
            print(f"No improvement: {self.best_rmse_linear:.4f} → {rmse_linear:.4f}. Reverting.")

            # === REVERT TO BEST KNOWN ===
            self.params = self.best_params.copy()
            self.filter_states = self.best_filter_states.copy()

            # === RESTORE GUI (revert) ===
            for i in range(self.n_filters):
                fc, Q, g = self.params[i*3:(i+1)*3]
                state = self.filter_states[i]

                self.line_edits[i*3].setText(f"{fc:.2f}")
                self.line_edits[i*3+1].setText(f"{Q:.3f}")
                self.line_edits[i*3+2].setText(f"{g:.2f}")

                self.dials[i*3].setValue(int(fc * 100))
                self.dials[i*3+1].setValue(int(Q * 1000))
                self.dials[i*3+2].setValue(int(g * 100))

                self.leds[i].setStyleSheet(
                    f"background-color: {'#44ff44' if state else '#ff4444'}; "
                    "border-radius: 6px; border: 1px solid #333;"
                )

                color = "#ff6666" if g < 0 else "#44ff88"
                self.gain_labels[i].setStyleSheet(f"font-weight: bold; color: {color};")
                self.line_edits[i*3+2].setStyleSheet(
                    f"background-color: #3a3a3a; border: 1px solid #7a5f1a; "
                    f"border-radius: 3px; color: {color};"
                )

                glow = "#ff4444" if g < 0 else "#44ff44"
                self.dials[i*3+2].glow_color = QColor(glow)
                self.dials[i*3+2].accent_color = QColor(glow)
                self.dials[i*3+2].update()

            # Recompute best curve
            H = total_response(self.params, self.z_full, self.fs, self.n_filters,
                               self.use_high_shelf, self.use_low_shelf, self.filter_states)
            mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
            corrected_db = self.meas_abs + mag_db
            error_db = corrected_db - self.target_abs

            self.eq_plot.setData(self.freqs_full, mag_db)
            self.corr_plot.setData(self.freqs_full, corrected_db)
            self.err_plot.setData(self.freqs_full, error_db)
            self.rmse_label.setText(f"RMSE: {self.best_rmse_linear:.4f} — {self.best_rmse_db:.4f} dB")
            QApplication.processEvents()
            print("GUI restored to best parameters and RMSE.")

    def closeEvent(self, event):
        # Recompute EQ response with best parameters
        H = total_response(
            self.best_params, self.z_full, self.fs, self.n_filters,
            self.use_high_shelf, self.use_low_shelf, self.best_filter_states
        )
        mag_linear = np.abs(H)
        mag_db = 20.0 * np.log10(np.maximum(mag_linear, 1e-12))

        # === 1. CORRECTED MEASUREMENT OUTPUT ===
        if self.corrected_output:
            corrected_db = self.meas_abs + mag_db
            df_corrected = pd.DataFrame({
                'frequency': self.freqs_full,
                'raw': corrected_db
            })
            try:
                df_corrected.to_csv(self.corrected_output, index=False)
                print(f"Corrected measurement (with best filters) written to {self.corrected_output}")
            except Exception as e:
                error_exit(f"Failed to write corrected measurement to {self.corrected_output}: {str(e)}")

        # === 2. RMSE CALCULATIONS ===
        # Fit region
        error_fit_linear = mag_linear[self.fit_mask] - self.desired_linear_fit
        error_fit_db     = mag_db[self.fit_mask] - self.desired_db_fit
        rms_fit_linear   = np.sqrt(np.mean(error_fit_linear**2))
        rms_fit_db       = np.sqrt(np.mean(error_fit_db**2))

        # Full band
        error_full_linear = mag_linear - (10**(self.target_abs/20.0))
        error_full_db     = mag_db - self.target_abs
        rms_full_linear   = np.sqrt(np.mean(error_full_linear**2))
        rms_full_db       = np.sqrt(np.mean(error_full_db**2))

        # === 3. PREAMP (based on peak in FIT REGION only) ===
        max_gain_fit = np.max(mag_db[self.fit_mask])
        preamp_db = -max_gain_fit

        # === 4. PRINT AND SAVE ===
        print("\nFinal adjusted filter parameters:")
        print(f"# Target: {self.target_file}")
        print(f"# Source: {self.source_file}")
        print(f"# RMS_error_linear_fit_region: {rms_fit_linear:.6f}")
        print(f"# RMS_error_dB_fit_region:     {rms_fit_db:.6f}")
        print(f"# RMS_error_linear_full_band:  {rms_full_linear:.6f}")
        print(f"# RMS_error_dB_full_band:      {rms_full_db:.6f}")
        print(f"Preamp: {preamp_db:.1f} dB")

        # Write to output file
        with open(self.output_file, 'w') as f:
            f.write(f"# Target: {self.target_file}\n")
            f.write(f"# Source: {self.source_file}\n")
            f.write(f"# RMS_error_linear_fit_region: {rms_fit_linear:.6f}\n")
            f.write(f"# RMS_error_dB_fit_region:     {rms_fit_db:.6f}\n")
            f.write(f"# RMS_error_linear_full_band:  {rms_full_linear:.6f}\n")
            f.write(f"# RMS_error_dB_full_band:      {rms_full_db:.6f}\n")
            f.write(f"Preamp: {preamp_db:.1f} dB\n")

            filters = pretty_filters_from_params(
                self.best_params, self.n_filters, self.best_filter_states, self.filter_types
            )
            for i, (t, fc, Q, g, state) in enumerate(filters):
                line = f"Filter {i+1}: {state} {t} Fc {fc:.2f} Hz Gain {g:.2f} dB Q {Q:.3f}\n"
                f.write(line)
                print(line.rstrip())

        event.accept()

def main():
    parser = argparse.ArgumentParser(
        description="Frequency response optimizer and EQ filter fitter.\n"
                    "Example: python acoustiq.py -s meas.csv -t target.csv -p 5 -hs 8000 -fbs '6000,12000' -co corrected.csv -fi filters.txt\n"
                    "This fits 5 peaking filters and a high-shelf at 8000 Hz with shelf bounds [6000, 12000] Hz,\n"
                    "optionally starting from filters in filters.txt, and writes the corrected measurement to corrected.csv."
    )

    parser.add_argument("-s", "--source", required=True, help="Source CSV (freq, dB)")
    parser.add_argument("-t", "--target", required=True, help="Target CSV (freq, dB)")
    parser.add_argument("-o", "--output", default="fitted_filters.txt", help="Output text file for filter parameters")
    parser.add_argument("-co", "--corrected-output", help="Output CSV file for corrected measurement (freq, dB)")
    parser.add_argument("-fi", "--filter-input", help="Input text file with initial filter parameters (e.g., fitted_filters.txt)")
    parser.add_argument("-f", "--freqs", help="Optional frequency grid (comma-separated values)")
    parser.add_argument("-m", "--maxiter", type=int, help=f"Maximum optimizer iterations (default: {OPT_SETTINGS['maxiter']})")
    parser.add_argument("-g", "--freq-grid-points", type=int, help=f"Number of frequency grid points (default: {OPT_SETTINGS['freq_grid_points']})")
    parser.add_argument("-r", "--fs", type=float, help=f"Sampling rate (Hz) (default: {OPT_SETTINGS['fs']})")
    parser.add_argument("-p", "--n-peaks", type=int, help=f"Number of peaking filters (default: {OPT_SETTINGS['n_peaks']})")
    parser.add_argument("-fb", "--fc-bounds", type=str, help="Global frequency bounds for all filters, e.g. '30,16000'")
    parser.add_argument("-fbp", "--fc-bounds-peaks", type=str, help="Frequency bounds for peaking filters, e.g. '20,1000'")
    parser.add_argument("-fbs", "--fc-bounds-shelf", type=str, help="Frequency bounds for shelf filters, e.g. '3000,20000'")
    parser.add_argument("-qb", "--Q-bounds", type=str, help="Q bounds for peaking filters, e.g. '0.3,8.0'")
    parser.add_argument("-sqb", "--shelf-q-bounds", type=str, help="Q bounds for shelf filters, e.g. '0.3,1.5'")
    parser.add_argument("-gb", "--gain-bounds", type=str, help="Gain bounds (dB), e.g. '-9,9'")
    parser.add_argument("-n1", "--no-anchor1k", action="store_true", help="Disable forcing filter stack to 0 dB at reference frequency")
    parser.add_argument("-aw", "--anchor-weight", type=float, default=50.0, help="Residual weight of the anchor constraint (default: 50.0)")
    parser.add_argument("-rf", "--ref-hz", type=float, default=1000.0, help="Reference frequency (Hz) for normalization (default: 1000)")
    parser.add_argument("-hs", "--high-shelf", type=float, help="Optional high-shelf center frequency (Hz). Adds one shelf filter optimized with peaks.")
    parser.add_argument("-ls", "--low-shelf", type=float, help="Optional low-shelf center frequency (Hz). Adds one shelf filter (first in the stack).")
    parser.add_argument("-sdo", "--shelf-drift-oct", type=float, default=0.3, help="Allowed ±octave drift for shelf center during optimization (default: 0.3)")
    parser.add_argument("-sw", "--shelf-weight", type=float, default=1.0, help="Weight multiplier for the shelf region (default: 1.0)")
    parser.add_argument("-flb", "--fc-bounds-low-shelf", type=str, help="Frequency bounds for low-shelf filter, e.g. '100,800'")
    parser.add_argument("-tl", "--tilt", type=float, default=0.0, help="Apply tilt (dB/oct) to target curve relative to ref_hz (default: 0.0)")
    parser.add_argument("-np", "--no-plot", action="store_true", help="Disable visualization")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive knob adjustment of filters (uses PyQt6)")

    args = parser.parse_args()

    if args.fs:
        OPT_SETTINGS["fs"] = args.fs
    if args.n_peaks is not None:
        OPT_SETTINGS["n_peaks"] = args.n_peaks
    if args.maxiter:
        OPT_SETTINGS["maxiter"] = args.maxiter
    if args.freq_grid_points:
        OPT_SETTINGS["freq_grid_points"] = args.freq_grid_points

    if args.fc_bounds:
        OPT_SETTINGS["bounds"]["fc"] = parse_bounds(args.fc_bounds)
    if args.Q_bounds:
        OPT_SETTINGS["bounds"]["Q"] = parse_bounds(args.Q_bounds)
    if args.gain_bounds:
        OPT_SETTINGS["bounds"]["gain_db"] = parse_bounds(args.gain_bounds)

    if args.shelf_q_bounds:
        shelf_Q_bounds = parse_bounds(args.shelf_q_bounds)
    else:
        shelf_Q_bounds = OPT_SETTINGS["bounds"]["shelf_Q"]

    print("\nEffective optimizer settings (after CLI overrides):")
    for k, v in OPT_SETTINGS.items():
        print(f"  {k}: {v}")
    print(f"  shelf_Q_bounds: {shelf_Q_bounds}")
    print("======================================================\n")

    base_n_peaks = OPT_SETTINGS["n_peaks"]
    use_high_shelf = args.high_shelf is not None
    n_filters = base_n_peaks + (1 if use_high_shelf else 0)
    use_low_shelf  = args.low_shelf  is not None
    n_filters += (1 if use_low_shelf else 0)

    # -----------------------------------------------------------------
    # 1. PEAK FILTER BOUNDS
    # -----------------------------------------------------------------
    fc_peaks_low  = OPT_SETTINGS["bounds"]["fc"][0]
    fc_peaks_high = OPT_SETTINGS["bounds"]["fc"][1]

    if args.fc_bounds_peaks:
        fc_peaks_low, fc_peaks_high = parse_bounds(args.fc_bounds_peaks)
        print(f"Peaking filters: user bounds [{fc_peaks_low:.1f}, {fc_peaks_high:.1f}] Hz")
    elif args.fc_bounds:
        fc_peaks_low, fc_peaks_high = parse_bounds(args.fc_bounds)
        print(f"Peaking filters: global bounds [{fc_peaks_low:.1f}, {fc_peaks_high:.1f}] Hz")

    # Clamp to valid audio range
    fc_peaks_low  = max(OPT_SETTINGS["bounds"]["fc"][0], fc_peaks_low)
    fc_peaks_high = min(OPT_SETTINGS["fs"]/2, fc_peaks_high)

    # -----------------------------------------------------------------
    # 2. LOW-SHELF BOUNDS (user-controlled or drift fallback)
    # -----------------------------------------------------------------
    fc_low_shelf_low  = fc_peaks_low
    fc_low_shelf_high = fc_peaks_high
    if use_low_shelf:
        fc_low_shelf = float(args.low_shelf)

        if args.fc_bounds_low_shelf:
            fc_low_shelf_low, fc_low_shelf_high = parse_bounds(args.fc_bounds_low_shelf)
            print(f"Low-shelf: user bounds [{fc_low_shelf_low:.1f}, {fc_low_shelf_high:.1f}] Hz")
        elif args.fc_bounds_shelf:
            fc_low_shelf_low, fc_low_shelf_high = parse_bounds(args.fc_bounds_shelf)
            print(f"Low-shelf: shared shelf bounds [{fc_low_shelf_low:.1f}, {fc_low_shelf_high:.1f}] Hz")
        else:
            drift_oct = args.shelf_drift_oct if args.shelf_drift_oct is not None else 0.3
            fc_low_shelf_low  = fc_low_shelf * (2.0 ** (-drift_oct))
            fc_low_shelf_high = fc_low_shelf * (2.0 ** (drift_oct))
            print(f"Low-shelf: drift ±{drift_oct:.2f} oct → [{fc_low_shelf_low:.1f}, {fc_low_shelf_high:.1f}] Hz")

        # Clamp to audio range
        fc_low_shelf_low  = max(OPT_SETTINGS["bounds"]["fc"][0], fc_low_shelf_low)
        fc_low_shelf_high = min(OPT_SETTINGS["fs"]/2, fc_low_shelf_high)

        # Clamp center
        if fc_low_shelf < fc_low_shelf_low or fc_low_shelf > fc_low_shelf_high:
            print(f"Warning: --low-shelf {fc_low_shelf:.1f} Hz outside bounds → clamping to [{fc_low_shelf_low:.1f}, {fc_low_shelf_high:.1f}]")
            fc_low_shelf = np.clip(fc_low_shelf, fc_low_shelf_low, fc_low_shelf_high)

    # -----------------------------------------------------------------
    # 3. HIGH-SHELF BOUNDS (user-controlled or drift fallback)
    # -----------------------------------------------------------------
    fc_shelf_low  = fc_peaks_low
    fc_shelf_high = fc_peaks_high
    if use_high_shelf:
        fc_shelf = float(args.high_shelf)

        if args.fc_bounds_shelf:
            fc_shelf_low, fc_shelf_high = parse_bounds(args.fc_bounds_shelf)
            print(f"High-shelf: user bounds [{fc_shelf_low:.1f}, {fc_shelf_high:.1f}] Hz")
        else:
            drift_oct = args.shelf_drift_oct if args.shelf_drift_oct is not None else 0.3
            fc_shelf_low  = fc_shelf * (2.0 ** (-drift_oct))
            fc_shelf_high = fc_shelf * (2.0 ** (drift_oct))
            print(f"High-shelf: drift ±{drift_oct:.2f} oct → [{fc_shelf_low:.1f}, {fc_shelf_high:.1f}] Hz")

        fc_shelf_low  = max(OPT_SETTINGS["bounds"]["fc"][0], fc_shelf_low)
        fc_shelf_high = min(OPT_SETTINGS["fs"]/2, fc_shelf_high)

        if fc_shelf < fc_shelf_low or fc_shelf > fc_shelf_high:
            print(f"Warning: --high-shelf {fc_shelf:.1f} Hz outside bounds → clamping to [{fc_shelf_low:.1f}, {fc_shelf_high:.1f}]")
            fc_shelf = np.clip(fc_shelf, fc_shelf_low, fc_shelf_high)

    # -----------------------------------------------------------------
    # 4. ENSURE BOUNDS ARE VALID (prevent scipy crash)
    # -----------------------------------------------------------------
    if use_low_shelf and fc_low_shelf_low >= fc_low_shelf_high - 1e-3:
        fc_low_shelf_high = fc_low_shelf_low + 1.0
        print("Warning: Low-shelf bounds collapsed → forcing 1 Hz width")

    if use_high_shelf and fc_shelf_low >= fc_shelf_high - 1e-3:
        fc_shelf_low = fc_shelf_high - 1.0
        print("Warning: High-shelf bounds collapsed → forcing 1 Hz width")

    try:
        df_meas = pd.read_csv(args.source)
        df_tgt = pd.read_csv(args.target)
    except pd.errors.EmptyDataError:
        error_exit("Input CSV file is empty or contains no valid data.")
    except FileNotFoundError as e:
        error_exit(f"Input CSV file not found: {str(e)}")
    except Exception as e:
        error_exit(f"Failed to read input CSV file: {str(e)}")

    if df_meas.shape[1] < 2 or df_tgt.shape[1] < 2:
        error_exit("Input CSV files must have at least two columns (frequency and dB).")

    try:
        fm = pd.to_numeric(df_meas.iloc[:, 0], errors='raise').to_numpy()
        ym = pd.to_numeric(df_meas.iloc[:, 1], errors='raise').to_numpy()
        ft = pd.to_numeric(df_tgt.iloc[:, 0], errors='raise').to_numpy()
        yt = pd.to_numeric(df_tgt.iloc[:, 1], errors='raise').to_numpy()
    except ValueError as e:
        error_exit("Input CSV files must contain numeric values in the first two columns (frequency and dB).")

    if not (np.all(np.isfinite(fm)) and np.all(np.isfinite(ym)) and np.all(np.isfinite(ft)) and np.all(np.isfinite(yt))):
        error_exit("Input CSV files must contain finite frequency and dB values.")

    if np.any(fm <= 0) or np.any(ft <= 0):
        error_exit("Input frequency arrays must contain strictly positive frequencies (Hz).")

    if args.freqs:
        freqs = np.array([float(x) for x in args.freqs.split(',')])
        if np.any(freqs <= 0) or np.any(freqs > OPT_SETTINGS["fs"] / 2.0) or not np.all(np.diff(freqs) > 0):
            error_exit("Custom frequency grid must contain positive, sorted frequencies within [0, fs/2].")
    else:
        freqs = np.logspace(np.log10(20.0), np.log10(min(OPT_SETTINGS["fs"] / 2.0, 20000.0)), OPT_SETTINGS["freq_grid_points"])

    freqs_full = np.asarray(freqs, dtype=float)
    freqs_full = freqs_full[np.isfinite(freqs_full)]
    freqs_full.sort()
    if freqs_full.size == 0:
        error_exit("Frequency grid is empty or invalid.")

    meas_interp = np.interp(np.log(freqs_full), np.log(fm), ym)
    tgt_interp = np.interp(np.log(freqs_full), np.log(ft), yt)

    tilt_slope = None
    if getattr(args, 'tilt', 0.0) != 0.0:
        ref_hz = args.ref_hz if args.ref_hz else 1000
        tilt_slope = args.tilt
        tilt_offset = tilt_slope * np.log2(freqs_full / ref_hz)
        tgt_interp += tilt_offset
        print(f"Applied target tilt: {tilt_slope:+.2f} dB/oct relative to {ref_hz} Hz")

    ref_hz = args.ref_hz
    meas_ref = np.interp(ref_hz, fm, ym)
    tgt_ref = np.interp(ref_hz, ft, yt)
    meas_interp -= meas_ref
    tgt_interp -= tgt_ref

    min_freq = float(np.nanmin(freqs_full))
    max_freq = float(np.nanmax(freqs_full))
    if not (np.isfinite(min_freq) and np.isfinite(max_freq) and min_freq > 0):
        error_exit("Invalid plotting frequency range.")

    if n_filters == 0:
        print("No filters defined (--n-peaks=0 and no --high-shelf). Outputting measurement vs. target comparison.")
        with open(args.output, "w") as fo:
            fo.write(f"# Target: {args.target}\n")
            fo.write(f"# Source: {args.source}\n")
            fo.write("# RMS_error_dB_fit_region: N/A (no filters applied)\n")
            fo.write("Preamp: 0.0 dB\n")
            fo.write("# No filters applied.\n")
        print(f"# Target: {args.target}")
        print(f"# Source: {args.source}")
        print("# RMS_error_dB_fit_region: N/A (no filters applied)")
        print("Preamp: 0.0 dB")
        print("# No filters applied.")

        if args.corrected_output:
            corrected_db = meas_interp
            df_corrected = pd.DataFrame({
                'frequency': freqs_full,
                'raw': corrected_db
            })
            try:
                df_corrected.to_csv(args.corrected_output, index=False)
                print(f"Corrected measurement written to {args.corrected_output}")
            except Exception as e:
                error_exit(f"Failed to write corrected measurement to {args.corrected_output}: {str(e)}")

        if not args.no_plot:
            meas_abs = meas_interp
            target_abs = tgt_interp
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs_full, y=meas_abs, mode='lines', name='Measurement',
                                    line=dict(color='deepskyblue')))
            fig.add_trace(go.Scatter(x=freqs_full, y=target_abs, mode='lines', name='Target',
                                    line=dict(color='gold')))
            title_text = "Measurement vs. Target (No EQ Applied)"
            if tilt_slope is not None:
                title_text += f", tilt {tilt_slope:.4f} dB/oct relative to {ref_hz} Hz"
            fig.update_layout(
                title=title_text,
                xaxis=dict(title='Frequency (Hz)', type='log', range=[math.log10(min_freq), math.log10(max_freq)], autorange=False),
                yaxis=dict(title='Amplitude (dB)'),
                legend=dict(x=1.05, y=1.05, xanchor='right', yanchor='top'),
                template="plotly_dark"
            )
            fig.show(config=dict(editable=True))
        else:
            print("Plotting disabled by --no-plot flag.")
        return

    fc_peaks_low, fc_peaks_high = OPT_SETTINGS["bounds"]["fc"]
    if args.fc_bounds_peaks:
        fc_peaks_low, fc_peaks_high = parse_bounds(args.fc_bounds_peaks)

    mask_peaks = (freqs_full >= fc_peaks_low) & (freqs_full <= fc_peaks_high)

    # low-shelf mask
    mask_low_shelf = np.zeros_like(freqs_full, dtype=bool)
    if use_low_shelf:
        mask_low_shelf = (freqs_full >= fc_low_shelf_low) & (freqs_full <= fc_low_shelf_high)

    # high-shelf mask
    mask_high_shelf = np.zeros_like(freqs_full, dtype=bool)
    if use_high_shelf:
        mask_high_shelf = (freqs_full >= fc_shelf_low) & (freqs_full <= fc_shelf_high)

    fit_mask = mask_peaks | mask_low_shelf | mask_high_shelf

    desired_db = tgt_interp - meas_interp
    desired_linear_full = 10**(desired_db/20.0)
    desired_linear_fit = desired_linear_full[fit_mask]

    freqs_fit = freqs_full[fit_mask]
    desired_db_fit = desired_db[fit_mask]

    # Convert the desired response to linear domain
    desired_linear_full = 10**(desired_db/20.0)
    desired_linear_fit = desired_linear_full[fit_mask]

    # Then continue with creating z_fit and z_full
    z_fit = np.exp(-1j * 2.0 * math.pi * freqs_fit / OPT_SETTINGS["fs"])
    z_full = np.exp(-1j * 2.0 * math.pi * freqs_full / OPT_SETTINGS["fs"])
    try:
        low_p, high_p = np.percentile(desired_db, [5.0, 95.0])
    except Exception:
        low_p, high_p = np.min(desired_db), np.max(desired_db)

    buffer_db = 3.0
    gain_min = math.floor((low_p - buffer_db) * 10.0) / 10.0
    gain_max = math.ceil((high_p + buffer_db) * 10.0) / 10.0
    gain_min = max(gain_min, -24.0)
    gain_max = min(gain_max, 24.0)
    OPT_SETTINGS["bounds"]["gain_db"] = (gain_min, gain_max)
    print(f"Adaptive gain bounds (percentile-based): ({gain_min:.2f}, {gain_max:.2f}) dB")

    if args.filter_input and args.interactive:
        p0, lb, ub, filter_states = parse_filter_file(args.filter_input, n_filters, use_high_shelf, use_low_shelf)
        print(f"\nLoaded filter parameters from {args.filter_input}:")
        for i in range(n_filters):
            fc, Q, g = p0[i*3:(i+1)*3]
            typ = 'LS' if (use_low_shelf and i == 0) else 'HS' if (use_high_shelf and i == n_filters-1) else 'PK'
            state = 'ON' if filter_states[i] else 'OFF'
            print(f"  Filter {i+1}: {typ} Fc {fc:.2f} Hz, Q {Q:.3f}, Gain {g:.2f} dB ({state})")
    else:
        guess_low = max(25.0, fc_peaks_low * 0.9)
        if base_n_peaks > 0:
            peak_fcs = np.exp(np.linspace(np.log(guess_low), np.log(fc_peaks_high), base_n_peaks))
            peak_gains = np.interp(peak_fcs, freqs_full, desired_db)
            peak_Qs = np.full(base_n_peaks, np.mean(OPT_SETTINGS["bounds"]["Q"]))
        else:
            peak_fcs = np.array([], dtype=float)
            peak_gains = np.array([], dtype=float)
            peak_Qs = np.array([], dtype=float)

        p0_list = []
        lb_list = []
        ub_list = []
        filter_states = [True] * n_filters

        # -----------------------------------------------------------------
        # 1. Low-shelf (index 0) - compute its own gain
        # -----------------------------------------------------------------
        if use_low_shelf:
            fc_center = float(args.low_shelf)
            mask = (freqs_full >= fc_low_shelf_low) & (freqs_full <= fc_low_shelf_high)
            if np.any(mask):
                meas_amp = 10 ** (meas_interp[mask] / 20.0)
                tgt_amp  = 10 ** (tgt_interp[mask]  / 20.0)
                desired_energy = (tgt_amp**2) / (meas_amp**2 + 1e-12)
                g_low = 20.0 * np.log10(np.sqrt(np.median(desired_energy)))
            else:
                g_low = np.interp(fc_center, freqs_full, desired_db)

            shelf_Q = 0.5 * (OPT_SETTINGS["bounds"]["shelf_Q"][0] + OPT_SETTINGS["bounds"]["shelf_Q"][1])
            p0_list.extend([fc_center, shelf_Q, g_low])
            lb_list.extend([fc_low_shelf_low, OPT_SETTINGS["bounds"]["shelf_Q"][0], gain_min])
            ub_list.extend([fc_low_shelf_high, OPT_SETTINGS["bounds"]["shelf_Q"][1], gain_max])

        # -----------------------------------------------------------------
        # 2. Peaking filters (middle)
        # -----------------------------------------------------------------
        for i in range(base_n_peaks):
            p0_list.extend([peak_fcs[i], peak_Qs[i], peak_gains[i]])
            lb_list.extend([fc_peaks_low, OPT_SETTINGS["bounds"]["Q"][0], OPT_SETTINGS["bounds"]["gain_db"][0]])
            ub_list.extend([fc_peaks_high, OPT_SETTINGS["bounds"]["Q"][1], OPT_SETTINGS["bounds"]["gain_db"][1]])

        # -----------------------------------------------------------------
        # 3. High-shelf (index n_filters-1) - compute its own gain
        # -----------------------------------------------------------------
        if use_high_shelf:
            fc_center = float(args.high_shelf)
            mask = (freqs_full >= fc_shelf_low) & (freqs_full <= fc_shelf_high)
            if np.any(mask):
                meas_amp = 10 ** (meas_interp[mask] / 20.0)
                tgt_amp  = 10 ** (tgt_interp[mask]  / 20.0)
                desired_energy = (tgt_amp**2) / (meas_amp**2 + 1e-12)
                g_high = 20.0 * np.log10(np.sqrt(np.median(desired_energy)))
            else:
                g_high = np.interp(fc_center, freqs_full, desired_db)

            shelf_Q = 0.5 * (OPT_SETTINGS["bounds"]["shelf_Q"][0] + OPT_SETTINGS["bounds"]["shelf_Q"][1])
            p0_list.extend([fc_center, shelf_Q, g_high])
            lb_list.extend([fc_shelf_low, OPT_SETTINGS["bounds"]["shelf_Q"][0], gain_min])
            ub_list.extend([fc_shelf_high, OPT_SETTINGS["bounds"]["shelf_Q"][1], gain_max])

        p0 = np.array(p0_list, dtype=float)
        lb = np.array(lb_list, dtype=float)
        ub = np.array(ub_list, dtype=float)

        if args.filter_input:
            p0_file, lb_file, ub_file, filter_states = parse_filter_file(args.filter_input, n_filters, use_high_shelf, use_low_shelf)
            print(f"\nUsing filter parameters from {args.filter_input} as initial guess:")
            for i in range(n_filters):
                fc, Q, g = p0_file[i*3:(i+1)*3]
                typ = 'LS' if (use_low_shelf and i == 0) else 'HS' if (use_high_shelf and i == n_filters-1) else 'PK'
                state = 'ON' if filter_states[i] else 'OFF'
                print(f"  Filter {i+1}: {typ} Fc {fc:.2f} Hz, Q {Q:.3f}, Gain {g:.2f} dB ({state})")
            p0 = p0_file
            lb = np.minimum(lb, lb_file)
            ub = np.maximum(ub, ub_file)

    print("\nInitial filter parameters (fc, Q, gain_db):")
    for i in range(n_filters):
        fc, Q, g = p0[i*3:(i+1)*3]
        typ = 'LS' if (use_low_shelf and i == 0) else 'HS' if (use_high_shelf and i == n_filters-1) else 'PK'
        state = 'ON' if filter_states[i] else 'OFF'
        print(f"  Filter {i+1}: {typ} Fc {fc:.2f} Hz, Q {Q:.3f}, Gain {g:.2f} dB ({state})")
    print("\nLower bounds:", lb)
    print("Upper bounds:", ub)

    anchor_enable = not args.no_anchor1k
    anchor_weight = float(args.anchor_weight)

    if p0.size > 0:
        p0 = np.clip(p0 + np.random.uniform(-1e-6, 1e-6, size=p0.shape), lb + 1e-12, ub - 1e-12)

    mask_peaks = (freqs_full >= fc_peaks_low) & (freqs_full <= fc_peaks_high)
    if use_high_shelf:
        mask_shelf = (freqs_full >= fc_shelf_low) & (freqs_full <= fc_shelf_high)
        fit_mask = mask_peaks | mask_shelf
    else:
        fit_mask = mask_peaks

    freqs_fit = freqs_full[fit_mask]
    desired_db_fit = desired_db[fit_mask]

    # Convert to linear domain for optimization
    desired_linear_fit = 10**(desired_db_fit/20.0)  # Convert target response to linear domain

    kernel_size = max(3, int(len(freqs_fit) * 0.03))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if freqs_fit.size == 0:
        error_msg = (f"No frequency points in fit union range (peaks [{fc_peaks_low}, {fc_peaks_high}] Hz"
                     + (f", shelf [{fc_shelf_low}, {fc_shelf_high}] Hz" if use_high_shelf else "") +
                     "). Check --freqs, --fc-bounds-peaks, --fc-bounds-shelf, or increase --freq-grid-points.")
        error_exit(error_msg)

    z_fit = np.exp(-1j * 2.0 * math.pi * freqs_fit / OPT_SETTINGS["fs"])
    z_full = np.exp(-1j * 2.0 * math.pi * freqs_full / OPT_SETTINGS["fs"])

    # Modify weight calculation for linear domain
    abs_dev_linear = np.abs(desired_linear_fit - 1.0)  # Deviation from unity gain
    max_abs_dev_linear = np.max(abs_dev_linear) if np.max(abs_dev_linear) > 0 else 1.0
    weight_fit = 1.0 + abs_dev_linear/max_abs_dev_linear
    weight_fit = smooth_vector(weight_fit, kernel_size)

    rolloff = 1.0 / (1.0 + (freqs_fit / fc_peaks_high)**4)
    # ---- low-shelf weighting ------------------------------------------------
    if use_low_shelf:
        low_shelf_mask_fit = (freqs_fit >= fc_low_shelf_low) & (freqs_fit <= fc_low_shelf_high)
        if np.any(low_shelf_mask_fit):
            rolloff[low_shelf_mask_fit] = 1.0
            weight_fit[low_shelf_mask_fit] *= args.shelf_weight * 5.0

    # ---- high-shelf weighting -----------------------------------------------
    if use_high_shelf:
        high_shelf_mask_fit = (freqs_fit >= fc_shelf_low) & (freqs_fit <= fc_shelf_high)
        if np.any(high_shelf_mask_fit):
            rolloff[high_shelf_mask_fit] = 1.0
            weight_fit[high_shelf_mask_fit] *= args.shelf_weight * 5.0
    weight_fit = np.maximum(weight_fit, 1e-6)

    if not (args.filter_input and args.interactive):
        res = least_squares(
            lambda x: residuals(x, z_fit, desired_linear_fit, n_filters,
                              weight_fit, anchor_enable, anchor_weight, ref_hz,
                              use_high_shelf, use_low_shelf, filter_states, OPT_SETTINGS["fs"])[0],
            p0,
            bounds=(lb, ub),
            max_nfev=OPT_SETTINGS["maxiter"],
            verbose=2,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            jac='3-point',
            loss='soft_l1',
            method='trf'
        )

        if not res.success:
            print(f"Warning: Optimizer did not converge: {res.message}. Results may be suboptimal.")
        p0 = res.x

    # --- BUILD filter_data LIKE GUI ---
    filter_data = []
    for i in range(n_filters):
        fc = p0[i*3]
        is_low_shelf = use_low_shelf and i == 0
        is_high_shelf = use_high_shelf and i == (n_filters - 1)
        typ = 'LS' if is_low_shelf else 'HS' if is_high_shelf else 'PK'
        filter_data.append((
            fc,
            p0[i*3:(i+1)*3],
            lb[i*3:(i+1)*3],
            ub[i*3:(i+1)*3],
            filter_states[i] if filter_states is not None else True,
            i,
            typ
        ))

    # Extract types
    filter_types = [f[6] for f in filter_data]

    # Now print
    filters = pretty_filters_from_params(p0, n_filters, filter_states, filter_types)

    p_opt = res.x
    fs = OPT_SETTINGS["fs"]

    H_opt = total_response(p_opt, z_full, fs, n_filters, use_high_shelf, use_low_shelf, filter_states)
    mag_opt_linear = np.abs(H_opt)
    mag_opt_db = 20.0 * np.log10(np.maximum(mag_opt_linear, 1e-12))

    meas_abs = np.interp(np.log(freqs_full), np.log(fm), ym) - meas_ref
    target_abs = tgt_interp
    eq_curve_db = mag_opt_db
    corrected_db = meas_abs + mag_opt_db
    error_curve_db = meas_abs - target_abs

    # Fit region RMSE
    rms_fit_linear = np.sqrt(np.mean((mag_opt_linear[fit_mask] - desired_linear_fit)**2))
    rms_fit_db = np.sqrt(np.mean((mag_opt_db[fit_mask] - desired_db_fit)**2))

    # Full band RMSE — USE target_abs FROM main()
    target_linear_full = 10**(target_abs / 20.0)  # ← NOW DEFINED
    rms_full_linear = np.sqrt(np.mean((mag_opt_linear - target_linear_full)**2))
    rms_full_db = np.sqrt(np.mean((mag_opt_db - target_abs)**2))

    print(f"\nOptimization done:")

    with open(args.output, "w") as fo:
        fo.write(f"# Target: {args.target}\n")
        fo.write(f"# Source: {args.source}\n")
        fo.write(f"# RMS_error_linear_fit_region: {rms_fit_linear:.6f}")
        fo.write(f"# RMS_error_dB_fit_region:     {rms_fit_db:.6f}")
        fo.write(f"# RMS_error_linear_full_band:  {rms_full_linear:.6f}")
        fo.write(f"# RMS_error_dB_full_band:      {rms_full_db:.6f}")
        print(f"# Target: {args.target}")
        print(f"# Source: {args.source}")
        print(f"# RMS_error_linear_fit_region: {rms_fit_linear:.6f}")
        print(f"# RMS_error_dB_fit_region:     {rms_fit_db:.6f}")
        print(f"# RMS_error_linear_full_band:  {rms_full_linear:.6f}")
        print(f"# RMS_error_dB_full_band:      {rms_full_db:.6f}")
        max_gain_db = np.max(mag_opt_db)
        preamp_db = -max_gain_db
        fo.write(f"Preamp: {preamp_db:.1f} dB\n")
        print(f"Preamp: {preamp_db:.1f} dB")
        for i, (typ, fc, Q, g, state) in enumerate(filters, start=1):
            line = f"Filter {i}: {state} {typ} Fc {fc:.2f} Hz Gain {g:.2f} dB Q {Q:.3f}\n"
            fo.write(line)
            print(line, end="")

    if not args.no_plot:
        if args.interactive and n_filters > 0:
            app = QApplication(sys.argv)
            window = EQWindow(
                p0, lb, ub, n_filters, freqs_full, z_full, meas_abs, target_abs,
                use_high_shelf, use_low_shelf, OPT_SETTINGS["fs"], args.corrected_output,
                desired_linear_fit, desired_db_fit,
                filter_states, z_fit, weight_fit, anchor_enable,
                anchor_weight, ref_hz, args.source, args.target,
                n_peaks=base_n_peaks,
                fc_bounds_peaks_str=args.fc_bounds_peaks,
                Q_bounds_str=args.Q_bounds,
                low_shelf_fc=args.low_shelf,
                high_shelf_fc=args.high_shelf,
                fc_bounds_low_shelf_str=args.fc_bounds_low_shelf,
                fc_bounds_shelf_str=args.fc_bounds_shelf,
                output_file=args.output,
                # PASS BOUNDS
                fc_peaks_low=fc_peaks_low,
                fc_peaks_high=fc_peaks_high,
                fc_low_shelf_low=fc_low_shelf_low if use_low_shelf else None,
                fc_low_shelf_high=fc_low_shelf_high if use_low_shelf else None,
                fc_shelf_low=fc_shelf_low if use_high_shelf else None,
                fc_shelf_high=fc_shelf_high if use_high_shelf else None
            )
            window.show()
            app.exec()
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs_full, y=meas_abs, mode='lines', name='Measurement',
                                    line=dict(color='deepskyblue')))
            fig.add_trace(go.Scatter(x=freqs_full, y=target_abs, mode='lines', name='Target',
                                    line=dict(color='gold')))
            fig.add_trace(go.Scatter(x=freqs_full, y=eq_curve_db, mode='lines', name='EQ Curve',
                                    line=dict(color='dodgerblue')))
            fig.add_trace(go.Scatter(x=freqs_full, y=corrected_db, mode='lines', name='Corrected Measurement',
                                    line=dict(color='lime', dash='dot')))
            fig.add_trace(go.Scatter(x=freqs_full, y=error_curve_db, mode='lines', name='Error Curve',
                                    line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([freqs_full, freqs_full[::-1]]),
                y=np.concatenate([target_abs, corrected_db[::-1]]),
                fill='toself',
                fillcolor='rgba(0,150,255,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='Residual'
            ))
            fig.add_vline(x=fc_peaks_low, line=dict(color='white', dash='dash'),
                        annotation_text=f"Peaks low: {fc_peaks_low:.0f} Hz", annotation_position="top left")
            fig.add_vline(x=fc_peaks_high, line=dict(color='white', dash='dash'),
                        annotation_text=f"Peaks high: {fc_peaks_high:.0f} Hz", annotation_position="top right")
            if use_low_shelf:
                fig.add_vline(x=fc_low_shelf_low, line=dict(color='lightblue', dash='dot'),
                              annotation_text=f"Low-shelf low: {fc_low_shelf_low:.0f} Hz",
                              annotation_position="bottom left")
                fig.add_vline(x=fc_low_shelf_high, line=dict(color='lightblue', dash='dot'),
                              annotation_text=f"Low-shelf high: {fc_low_shelf_high:.0f} Hz",
                              annotation_position="bottom right")
            if use_high_shelf:
                fig.add_vline(x=fc_shelf_low, line=dict(color='lightgreen', dash='dot'),
                            annotation_text=f"Shelf low: {fc_shelf_low:.0f} Hz", annotation_position="bottom left")
                fig.add_vline(x=fc_shelf_high, line=dict(color='lightgreen', dash='dot'),
                            annotation_text=f"Shelf high: {fc_shelf_high:.0f} Hz", annotation_position="bottom right")

            title_text = f"EQ Fit vs Target (peaks {fc_peaks_low:.0f}-{fc_peaks_high:.0f} Hz"
            if use_high_shelf:
                title_text += f", shelf {fc_shelf_low:.0f}-{fc_shelf_high:.0f} Hz"
            if tilt_slope is not None:
                title_text += f", tilt {tilt_slope:.4f} dB/oct relative to {ref_hz} Hz"
            title_text += ")"

            fig.update_layout(
                title=title_text,
                xaxis=dict(title='Frequency (Hz)', type='log', range=[math.log10(min_freq), math.log10(max_freq)], autorange=False),
                yaxis=dict(title='Amplitude (dB)'),
                legend=dict(x=1.05, y=1.05, xanchor='right', yanchor='top'),
                template="plotly_dark"
            )

            fig.show(config=dict(editable=True))
    else:
        print("Plotting disabled by --no-plot flag.")

if __name__ == "__main__":
    main()