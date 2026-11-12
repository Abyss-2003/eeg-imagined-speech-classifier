"""
EEG Imagined Speech — Live Demo UI
====================================
Run with:
    streamlit run app.py
"""

import os
import io
import tempfile
import numpy as np
import joblib
import mne
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import welch
from collections import Counter
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Mind Reader",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8ff;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] .stMarkdown { color: #c8c8e8; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}

/* ── Prediction badge ── */
.prediction-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 20px 60px rgba(102,126,234,0.4); }
    50%  { box-shadow: 0 20px 80px rgba(102,126,234,0.7); }
    100% { box-shadow: 0 20px 60px rgba(102,126,234,0.4); }
}
.digit-display {
    font-size: 120px;
    font-weight: 900;
    color: white;
    line-height: 1;
    text-shadow: 0 0 40px rgba(255,255,255,0.5);
}
.confidence-text {
    font-size: 18px;
    color: rgba(255,255,255,0.85);
    margin-top: 8px;
}

/* ── Metric tiles ── */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section headers ── */
.section-title {
    font-size: 13px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

/* ── Progress bar colour ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
    background: rgba(167,139,250,0.05) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-weight: 600;
    font-size: 16px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.5);
}

/* hide default header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5",
    "T7",  "P7", "O1", "O2",
    "P8",  "T8", "FC6", "F4",
    "F8",  "AF4"
]

BANDS = {
    "Delta\n1–4 Hz":  (1,  4),
    "Theta\n4–8 Hz":  (4,  8),
    "Alpha\n8–13 Hz": (8, 13),
    "Beta\n13–30 Hz": (13, 30),
    "Gamma\n30–45 Hz":(30, 45),
}

DIGIT_EMOJI = {
    0: "0️⃣", 1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣",
    5: "5️⃣", 6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣",
}

MODEL_PATHS = {
    "Enhanced (Band Powers + Hjorth + ZCR)": "eeg_digit_enhanced.joblib",
    "Paper Baseline (SD + RMS + Sum + Energy)": "eeg_digit_rf.joblib",
}


# ─────────────────────────────────────────────────────────────
# SIGNAL PROCESSING  (identical to training pipeline)
# ─────────────────────────────────────────────────────────────

def moving_average(signal, window=5):
    kernel   = np.ones(window) / window
    smoothed = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        smoothed[i] = np.convolve(signal[i], kernel, mode="same")
    return smoothed


def paper_features(seg):
    is_2d = (seg.ndim == 2)
    if is_2d:
        seg = seg[np.newaxis, ...]
    T = seg.shape[2]
    ddof = 1 if T > 1 else 0
    sd   = np.std(seg, axis=2, ddof=ddof)
    rms  = np.sqrt(np.mean(seg**2, axis=2))
    s    = np.sum(seg, axis=2)
    e    = np.sum(seg**2, axis=2)
    out  = np.concatenate([sd, rms, s, e], axis=1)
    return out[0] if is_2d else out


def band_powers(seg, sr=128):
    is_2d = (seg.ndim == 2)
    if is_2d:
        seg = seg[np.newaxis, ...]
    M, n_channels, T = seg.shape
    nperseg = min(T, T)
    freqs, psd = welch(seg, fs=sr, nperseg=nperseg, noverlap=0, scaling="density", axis=-1)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    band_list = []
    for _, (fl, fh) in BANDS.items():
        mask = (freqs >= fl) & (freqs < fh)
        if mask.any():
            p_band = np.sum(psd[:, :, mask], axis=-1) * df
        else:
            p_band = np.zeros((M, n_channels))
        band_list.append(p_band)
    out = np.stack(band_list, axis=2).reshape(M, -1)
    return out[0] if is_2d else out


def hjorth(seg):
    is_2d = (seg.ndim == 2)
    if is_2d:
        seg = seg[np.newaxis, ...]
    M, n_channels, T = seg.shape
    dx  = np.diff(seg, axis=2)
    ddx = np.diff(dx, axis=2)
    vx  = np.var(seg, axis=2, ddof=1) if T > 1 else np.full((M, n_channels), 1e-12)
    vdx = np.var(dx,  axis=2, ddof=1) if dx.shape[2] > 1 else np.full((M, n_channels), 1e-12)
    vddx= np.var(ddx, axis=2, ddof=1) if ddx.shape[2] > 1 else np.full((M, n_channels), 1e-12)
    mob  = np.sqrt(vdx  / np.maximum(vx,  1e-12))
    comp = np.sqrt(vddx / np.maximum(vdx, 1e-12)) / np.maximum(mob, 1e-12)
    out = np.concatenate([vx, mob, comp], axis=1)
    return out[0] if is_2d else out


def zcr(seg):
    is_2d = (seg.ndim == 2)
    if is_2d:
        seg = seg[np.newaxis, ...]
    T = seg.shape[2]
    signs = np.sign(seg)
    cross = np.diff(signs, axis=2) != 0
    out = np.sum(cross, axis=2) / max(T - 1, 1)
    return out[0] if is_2d else out


def extract_features_enhanced(seg, sr=128):
    is_2d = (seg.ndim == 2)
    if is_2d:
        seg = seg[np.newaxis, ...]
    out = np.concatenate([
        paper_features(seg),
        band_powers(seg, sr),
        hjorth(seg),
        zcr(seg)
    ], axis=1)
    return out[0] if is_2d else out


def extract_features_paper(seg, sr=128):
    return paper_features(seg)


def get_windows(data, sr, window_s=0.050, hop_s=0.025):
    signal = moving_average(data)
    win    = int(round(window_s * sr))
    hop    = int(round(hop_s   * sr))
    N      = signal.shape[1]
    windows = []
    idx = 0
    while idx + win <= N:
        windows.append(signal[:, idx: idx + win])
        idx += hop
    return windows


def majority_vote(preds):
    c = Counter(preds)
    m = max(c.values())
    return min(k for k, v in c.items() if v == m)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# EDF LOADER
# ─────────────────────────────────────────────────────────────

def load_edf_bytes(file_bytes):
    """Load EDF from uploaded bytes, return (data, sr, ch_names)."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
        missing = [c for c in EEG_CHANNELS if c not in raw.ch_names]
        if missing:
            raise ValueError(f"Missing channels: {missing}")
        raw.pick(EEG_CHANNELS)
        data = raw.get_data()
        sr   = float(raw.info["sfreq"])
    finally:
        os.unlink(tmp_path)
    return data, sr


# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────

PLOT_BG   = "rgba(0,0,0,0)"
PLOT_FONT = dict(color="#c8c8e8", family="Inter")
GRID_COLOR = "rgba(255,255,255,0.08)"


def eeg_plot(data, sr, title="Raw EEG Signal (all 14 channels)"):
    T      = data.shape[1]
    t      = np.linspace(0, T / sr, T)
    colors = px.colors.sequential.Plasma_r

    fig = go.Figure()
    offset = 0
    spacing = np.max(np.abs(data)) * 2.5

    for i, ch in enumerate(EEG_CHANNELS):
        fig.add_trace(go.Scatter(
            x=t, y=data[i] + i * spacing,
            mode="lines",
            line=dict(width=0.9, color=colors[i % len(colors)]),
            name=ch,
            hovertemplate=f"<b>{ch}</b><br>t=%{{x:.3f}}s<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, **PLOT_FONT)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        xaxis=dict(title="Time (s)", gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(
            tickvals=[i * spacing for i in range(14)],
            ticktext=EEG_CHANNELS,
            gridcolor=GRID_COLOR, showgrid=True,
        ),
        height=460,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        margin=dict(l=60, r=10, t=40, b=10),
    )
    return fig


def confidence_bar(probs):
    colors = [
        "#667eea" if i == np.argmax(probs) else "rgba(167,139,250,0.35)"
        for i in range(10)
    ]
    fig = go.Figure(go.Bar(
        x=list(range(10)),
        y=probs,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#c8c8e8"),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(10)),
            ticktext=[str(d) for d in range(10)],
            title="Digit",
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(title="Confidence", range=[0, max(probs) * 1.25],
                   gridcolor=GRID_COLOR),
        height=280,
        margin=dict(l=40, r=10, t=10, b=40),
        bargap=0.25,
    )
    return fig


def band_power_heatmap(data, sr):
    """Heatmap: 14 channels × 5 bands of average band power."""
    windows = get_windows(data, sr)
    band_names = list(BANDS.keys())
    matrix = np.zeros((14, 5))

    for seg in windows:
        bp = band_powers(seg, sr).reshape(14, 5)
        matrix += bp

    matrix /= max(len(windows), 1)
    matrix_db = 10 * np.log10(matrix + 1e-30)   # convert to dB

    fig = go.Figure(go.Heatmap(
        z=matrix_db,
        x=band_names,
        y=EEG_CHANNELS,
        colorscale="Plasma",
        colorbar=dict(title="Power (dB)", thickness=12,
                      tickfont=dict(color="#c8c8e8")),
        hovertemplate="Channel: %{y}<br>Band: %{x}<br>Power: %{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Frequency Band Power Heatmap", font=dict(size=14, **PLOT_FONT)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        height=380,
        margin=dict(l=60, r=10, t=40, b=10),
        xaxis=dict(tickangle=-15),
    )
    return fig


def window_prediction_timeline(window_preds):
    """Show how window predictions evolve across the trial."""
    x = list(range(len(window_preds)))
    fig = go.Figure()

    # Scatter dots coloured by predicted digit
    fig.add_trace(go.Scatter(
        x=x,
        y=window_preds,
        mode="markers+lines",
        marker=dict(
            size=6,
            color=window_preds,
            colorscale="Plasma",
            showscale=False,
        ),
        line=dict(color="rgba(167,139,250,0.3)", width=1),
        hovertemplate="Window %{x}: Digit %{y}<extra></extra>",
        name="Window prediction",
    ))

    # Majority vote line
    vote = majority_vote(window_preds)
    fig.add_hline(
        y=vote,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"  Final prediction: {vote}",
        annotation_font_color="#f59e0b",
    )

    fig.update_layout(
        title=dict(text="Window-by-Window Predictions", font=dict(size=14, **PLOT_FONT)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=PLOT_FONT,
        xaxis=dict(title="Window index", gridcolor=GRID_COLOR),
        yaxis=dict(
            title="Predicted digit",
            tickmode="array",
            tickvals=list(range(10)),
            ticktext=[str(d) for d in range(10)],
            gridcolor=GRID_COLOR,
        ),
        height=260,
        margin=dict(l=50, r=10, t=40, b=40),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <div style='font-size:48px'>🧠</div>
        <div style='font-size:20px; font-weight:700; color:white'>EEG Mind Reader</div>
        <div style='font-size:12px; color:#a78bfa; margin-top:4px'>
            Kumar et al. 2017 — Imagined Speech
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1)'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select model",
        list(MODEL_PATHS.keys()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-title" style="margin-top:20px">📂 Upload EEG</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop an .edf file here",
        type=["edf"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:rgba(255,255,255,0.1)">', unsafe_allow_html=True)

    # Quick-demo: pick from existing dataset files
    st.markdown('<div class="section-title">🎲 Quick Demo</div>', unsafe_allow_html=True)
    demo_digit = st.selectbox("Pick any digit to demo with", list(range(10)),
                              format_func=lambda x: f"Digit {x}  {DIGIT_EMOJI[x]}")
    demo_subj  = st.selectbox("Pick any participant", list(range(0, 25)),
                               format_func=lambda x: f"Participant {x}")
    use_demo   = st.button("🚀 Run Demo Prediction")

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.1)'>
    <div style='font-size:11px; color:rgba(255,255,255,0.3); text-align:center'>
        14-channel Emotiv EPOC+ · 128 Hz · Random Forest<br>
        <b style='color:#a78bfa'>~70.4% window accuracy</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN PAGE HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding:30px 0 10px'>
    <div style='font-size:13px; font-weight:600; color:#a78bfa;
                letter-spacing:3px; text-transform:uppercase'>
        Brain–Computer Interface Demo
    </div>
    <div style='font-size:40px; font-weight:900; color:white; margin-top:6px'>
        What digit are you thinking of?
    </div>
    <div style='font-size:16px; color:rgba(255,255,255,0.5); margin-top:8px'>
        Upload a raw EEG recording and the AI will decode the imagined digit
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# RESOLVE INPUT  (uploaded file OR quick demo)
# ─────────────────────────────────────────────────────────────

edf_bytes    = None
ground_truth = None
source_label = ""

if uploaded is not None:
    edf_bytes    = uploaded.read()
    source_label = f"📄 {uploaded.name}"

elif use_demo:
    # Try to find the demo file from the dataset
    demo_candidates = [
        f"./data/Digit/name{demo_subj}_{demo_digit}.edf",
        f"./data/Digit/name{demo_subj+1}_{demo_digit}.edf",
    ]
    demo_path = None
    for p in demo_candidates:
        if os.path.exists(p):
            demo_path = p
            break

    if demo_path:
        with open(demo_path, "rb") as f:
            edf_bytes = f.read()
        ground_truth = demo_digit
        source_label = f"📄 {os.path.basename(demo_path)}  (ground truth: {demo_digit})"
    else:
        st.warning(
            f"Demo file for participant {demo_subj} / digit {demo_digit} "
            "not found. Try a different combination."
        )


# ─────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────

if edf_bytes is not None:

    # ── Load model ──
    model_path   = MODEL_PATHS[model_choice]
    is_enhanced  = "Enhanced" in model_choice

    with st.spinner("Loading model…"):
        artifact = load_model(model_path)

    if artifact is None:
        st.error(
            f"Model file `{model_path}` not found. "
            "Please train the model first by running `eeg_digit_rf.py` or "
            "`eeg_digit_enhanced.py`."
        )
        st.stop()

    model = artifact["model"]

    # ── Load EDF ──
    with st.spinner("Reading EEG recording…"):
        try:
            data, sr = load_edf_bytes(edf_bytes)
        except Exception as e:
            st.error(f"Error loading EDF: {e}")
            st.stop()

    duration = data.shape[1] / sr

    # ── Extract windows & features ──
    with st.spinner("Extracting features from EEG windows…"):
        windows = get_windows(data, sr,
                              window_s=0.050,
                              hop_s=0.025 if is_enhanced else 0.050)

        feat_fn = extract_features_enhanced if is_enhanced else extract_features_paper

        windows_arr = np.array(windows)
        X = feat_fn(windows_arr, sr)

    # ── Predict ──
    with st.spinner("Running Random Forest classifier…"):
        time.sleep(0.3)   # brief pause so the spinner is visible
        window_preds = model.predict(X)
        window_probs = model.predict_proba(X)

        avg_probs    = window_probs.mean(axis=0)
        final_digit  = majority_vote(window_preds.tolist())
        confidence   = avg_probs[final_digit] * 100

    # ───────────────────────────────────────────
    # LAYOUT  —  prediction + charts
    # ───────────────────────────────────────────

    # Row 1: Prediction card  +  quick stats
    col_pred, col_stats = st.columns([1, 1], gap="large")

    with col_pred:
        correct_flag = ""
        if ground_truth is not None:
            correct_flag = (
                "  ✅ Correct!" if final_digit == ground_truth
                else f"  ❌ True: {ground_truth}"
            )

        st.markdown(f"""
        <div class="prediction-box">
            <div style='font-size:14px; color:rgba(255,255,255,0.7);
                        letter-spacing:2px; text-transform:uppercase;
                        margin-bottom:8px'>
                Predicted Digit
            </div>
            <div class="digit-display">{final_digit}</div>
            <div style='font-size:40px; margin-top:4px'>{DIGIT_EMOJI[final_digit]}</div>
            <div class="confidence-text">
                Confidence: <b>{confidence:.1f}%</b>{correct_flag}
            </div>
            <div style='font-size:12px; color:rgba(255,255,255,0.45);
                        margin-top:12px'>{source_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="section-title">Recording Details</div>',
                    unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{duration:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{int(sr)} Hz</div>
                <div class="metric-label">Sample Rate</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{len(windows)}</div>
                <div class="metric-label">Windows</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{X.shape[1]}</div>
                <div class="metric-label">Features</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px">Confidence per Digit</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(confidence_bar(avg_probs), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("---")

    # Row 2: EEG signal  +  window timeline
    col_eeg, col_time = st.columns([3, 2], gap="large")

    with col_eeg:
        st.plotly_chart(eeg_plot(data, sr), use_container_width=True,
                        config={"displayModeBar": False})

    with col_time:
        st.plotly_chart(window_prediction_timeline(window_preds.tolist()),
                        use_container_width=True,
                        config={"displayModeBar": False})

        # Vote breakdown
        vote_counts = Counter(window_preds.tolist())
        st.markdown('<div class="section-title">Vote Breakdown</div>',
                    unsafe_allow_html=True)
        total = len(window_preds)
        for digit in sorted(vote_counts, key=lambda d: -vote_counts[d])[:5]:
            pct = vote_counts[digit] / total
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"color:#c8c8e8;font-size:13px;margin-bottom:4px'>"
                f"<span>Digit {digit} {DIGIT_EMOJI[digit]}</span>"
                f"<span>{vote_counts[digit]} / {total} &nbsp;({pct:.0%})</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)

    st.markdown("---")

    # Row 3: Band power heatmap  (enhanced only)
    if is_enhanced:
        st.plotly_chart(band_power_heatmap(data, sr), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Top-3 candidates ──
    top3_idx  = np.argsort(avg_probs)[::-1][:3]
    st.markdown('<div class="section-title">Top 3 Candidates</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, idx) in enumerate(zip(cols, top3_idx)):
        with col:
            st.markdown(f"""
            <div class="metric-tile" style="text-align:center;padding:20px">
                <div style='font-size:28px'>{medals[i]}</div>
                <div style='font-size:48px;font-weight:900;color:white'>{idx}</div>
                <div style='font-size:14px;color:#a78bfa;margin-top:4px'>
                    {DIGIT_EMOJI[idx]}
                </div>
                <div style='font-size:20px;font-weight:700;color:white;margin-top:4px'>
                    {avg_probs[idx]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── EMPTY STATE ──
    st.markdown("""
    <div style='text-align:center; padding:60px 20px'>
        <div style='font-size:80px'>🧠</div>
        <div style='font-size:22px; color:rgba(255,255,255,0.6); margin-top:16px'>
            Upload an <b style='color:#a78bfa'>.edf</b> file from the sidebar,<br>
            or use the <b style='color:#a78bfa'>Quick Demo</b> to pick any recording
            from the dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show model status
    st.markdown("---")
    st.markdown('<div class="section-title">Model Status</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for col, (name, path) in zip(cols, MODEL_PATHS.items()):
        exists = os.path.exists(path)
        with col:
            st.markdown(f"""
            <div class="metric-tile" style="text-align:left">
                <div style='font-size:24px'>{"✅" if exists else "❌"}</div>
                <div style='font-size:13px;color:white;margin-top:6px;font-weight:600'>
                    {name}
                </div>
                <div style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:4px'>
                    {path}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # How it works
    st.markdown("---")
    st.markdown('<div class="section-title">How It Works</div>',
                unsafe_allow_html=True)

    steps = [
        ("📡", "Record",   "14-channel EEG via Emotiv EPOC+ at 128 Hz while imagining a digit"),
        ("🔧", "Filter",   "5-point moving average smooths high-frequency noise"),
        ("✂️",  "Slice",    "Recording split into 50ms windows (~6 samples each)"),
        ("📊", "Features", "56–182 statistical & spectral features extracted per window"),
        ("🌲", "Classify", "Random Forest (40 trees) votes on each window"),
        ("🗳️",  "Vote",     "Majority vote across all windows → final digit prediction"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="metric-tile" style="text-align:center;min-height:150px">
                <div style='font-size:28px'>{icon}</div>
                <div style='font-size:13px;font-weight:700;color:#a78bfa;margin-top:8px'>
                    {title}
                </div>
                <div style='font-size:11px;color:rgba(255,255,255,0.5);margin-top:6px'>
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
