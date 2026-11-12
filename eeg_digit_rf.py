
import os
import re
import glob
import joblib
import mne
import numpy as np

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "./data/Digit"

EXPECTED_CHANNELS = 14
EXPECTED_SAMPLING_RATE = 128

# Paper: 5-point moving average filter (Section 3.1)
SMOOTHING_WINDOW = 5

# Fine-level (digit) classification: 50ms windows (Section 3.2)
# At 128 Hz: 50ms = 6.4 ≈ 6 samples
# The classifier is trained/tested at window level (not trial level)
SEGMENT_DURATION = 0.050   # 50 milliseconds

# Paper: Random Forest with 40 trees (Section 3.3)
N_TREES = 40

# Paper: 10-fold cross-validation (within-subject / subject-dependent)
N_FOLDS = 10

RANDOM_STATE = 42


# ============================================================
# 1. EXTRACT PARTICIPANT ID AND DIGIT LABEL
# ============================================================

def extract_metadata(filename):

    name = os.path.basename(filename)

    pattern = r"name(\d+)_(\d+)\.edf"

    match = re.fullmatch(pattern, name, re.IGNORECASE)

    if not match:
        raise ValueError("Invalid filename: " + name)

    participant = int(match.group(1))
    label = int(match.group(2))

    if label < 0 or label > 9:
        raise ValueError("Invalid digit label: " + name)

    return participant, label


# ============================================================
# EEG CHANNEL CONFIGURATION
# ============================================================

# Emotiv EPOC+ 14 EEG electrodes (paper uses all 14)
EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5",
    "T7",  "P7", "O1", "O2",
    "P8",  "T8", "FC6", "F4",
    "F8",  "AF4"
]


# ============================================================
# LOAD EDF FILE
# ============================================================

def load_edf(filepath):

    raw = mne.io.read_raw_edf(
        filepath,
        preload=True,
        verbose="ERROR"
    )

    # Verify all required electrodes exist
    missing_channels = [
        ch for ch in EEG_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f"Missing EEG channels: {missing_channels}"
        )

    # Select ONLY the 14 EEG electrodes used in the paper
    raw.pick(EEG_CHANNELS)

    # Extract EEG data as (14, samples)
    data = raw.get_data()

    sampling_rate = float(raw.info["sfreq"])

    # Validate channel count
    assert data.shape[0] == 14

    # Validate sampling frequency
    if not np.isclose(sampling_rate, 128.0):
        raise ValueError(
            f"Expected 128 Hz, found {sampling_rate}"
        )

    return data, sampling_rate


# ============================================================
# 3. MOVING AVERAGE FILTER  (Section 3.1 of the paper)
# ============================================================

def moving_average(signal, window=5):
    """Apply a 5-point moving average along the time axis (per channel)."""

    channels, samples = signal.shape

    smoothed = np.zeros(
        (channels, samples),
        dtype=np.float64
    )

    kernel = np.ones(window) / window

    for i in range(channels):
        smoothed[i] = np.convolve(
            signal[i],
            kernel,
            mode="same"
        )

    return smoothed


# ============================================================
# 4. FEATURE EXTRACTION  (Section 3.2 of the paper)
#
#    Paper extracts 4 time-domain features per channel:
#      SD, RMS, Sum-of-Values, Energy
#    → 14 channels × 4 features = 56-dim feature vector per window
# ============================================================

def extract_features(segment):
    """
    Compute the four statistical time-domain features described in the paper.

    Parameters
    ----------
    segment : ndarray, shape (14, T)
        One windowed EEG segment across all 14 channels.

    Returns
    -------
    ndarray, shape (56,)
        Concatenated [SD, RMS, Sum, Energy] for each channel.
    """

    # Standard Deviation
    # Use ddof=0 (population SD) for very short windows to avoid division
    # issues when T is small (the paper does not specify ddof)
    if segment.shape[1] > 1:
        sd = np.std(segment, axis=1, ddof=1)
    else:
        sd = np.zeros(segment.shape[0])

    # Root Mean Square
    rms = np.sqrt(np.mean(segment ** 2, axis=1))

    # Sum of Values
    summation = np.sum(segment, axis=1)

    # Signal Energy
    energy = np.sum(segment ** 2, axis=1)

    # 14 + 14 + 14 + 14 = 56 features
    features = np.concatenate([sd, rms, summation, energy])

    return features


# ============================================================
# 5. SEGMENT EEG RECORDING INTO 50ms WINDOWS
#    (Section 3.2 of the paper — fine-level digit classification)
#
#    Each window is classified independently; accuracy is computed
#    at the window level within the 10-fold CV.
# ============================================================

def extract_recording_features(signal, sampling_rate):
    """
    Slice a recording into non-overlapping 50ms windows,
    apply the moving-average filter, then extract the 56-dim feature
    vector from each window.

    Parameters
    ----------
    signal : ndarray, shape (14, N)
    sampling_rate : float  (128 Hz)

    Returns
    -------
    ndarray, shape (n_windows, 56)
        Each row is one 50ms window feature vector.
    """

    # Step 1: apply the 5-point moving average
    signal = moving_average(signal, SMOOTHING_WINDOW)

    # 50ms at 128 Hz = 6.4 samples → 6 samples
    # We use time-based rounding to match the paper's approach
    window_seconds = SEGMENT_DURATION

    total_duration = signal.shape[1] / sampling_rate

    n_windows = int(np.floor(total_duration / window_seconds))

    features = []

    for i in range(n_windows):

        start = int(round(i * window_seconds * sampling_rate))
        end   = int(round((i + 1) * window_seconds * sampling_rate))

        segment = signal[:, start:end]

        if segment.shape[1] < 2:
            continue

        feature_vector = extract_features(segment)

        features.append(feature_vector)

    if len(features) == 0:
        raise ValueError("No valid EEG windows generated")

    return np.array(features)   # shape: (n_windows, 56)


# ============================================================
# 6. LOAD COMPLETE DATASET
# ============================================================

def load_dataset():

    files = sorted(
        glob.glob(os.path.join(DATASET_PATH, "*.edf"))
    )

    if len(files) == 0:
        raise ValueError("No EDF files found")

    records = []

    skipped = 0

    print("\nLoading EEG recordings...\n")

    for filepath in files:

        try:

            participant, label = extract_metadata(filepath)

            eeg, sampling_rate = load_edf(filepath)

            features = extract_recording_features(
                eeg,
                sampling_rate
            )

            records.append({

                "participant": participant,

                "label": label,

                "features": features,   # shape: (n_windows, 56)

                "filename": os.path.basename(filepath)

            })

        except Exception as e:

            skipped += 1

            print(
                f"Skipping {os.path.basename(filepath)}: {e}"
            )

    if len(records) == 0:
        raise ValueError("No valid recordings loaded")

    print("\n========== DATASET SUMMARY ==========")

    print("Total EDF files:", len(files))
    print("Successfully loaded:", len(records))
    print("Skipped:", skipped)

    unique_participants = sorted(set(r["participant"] for r in records))
    print("Unique participants:", len(unique_participants))

    total_windows = sum(len(r["features"]) for r in records)
    print("Total windows:", total_windows)
    print(
        "Windows per recording (sample):",
        records[0]["features"].shape[0]
    )

    print("\nDigit distribution:")

    for digit in range(10):

        count = sum(r["label"] == digit for r in records)

        print(f"  Digit {digit}: {count} recordings")

    return records


# ============================================================
# 7. MAJORITY VOTING (used only for trial-level reporting)
# ============================================================

def majority_vote(predictions):
    """Return the most common class; ties broken by smallest digit."""

    counts = Counter(predictions)

    max_count = max(counts.values())

    candidates = [
        digit
        for digit, count in counts.items()
        if count == max_count
    ]

    return min(candidates)


# ============================================================
# 8. RANDOM FOREST MODEL  (Section 3.3 of the paper)
# ============================================================

def create_model(random_state=RANDOM_STATE):

    model = RandomForestClassifier(

        n_estimators=N_TREES,

        criterion="gini",

        bootstrap=True,

        max_features="sqrt",

        random_state=random_state,

        n_jobs=-1

    )

    return model


# ============================================================
# 9. WITHIN-SUBJECT 10-FOLD CROSS-VALIDATION
#
#    The paper evaluates at the WINDOW level (fine-level classification).
#    For each participant:
#      1. Pool all their 50ms windows (from all 10 digit recordings).
#      2. Stratified 10-fold split on windows.
#      3. Train RF on 9/10 of windows, test on 1/10.
#      4. Report window-level accuracy.
#
#    Mean accuracy averaged across all subjects = final score.
# ============================================================

def evaluate_model(records):
    """
    Within-subject (subject-dependent) 10-fold cross-validation
    at the window level, matching the paper's fine-level evaluation.
    """

    unique_participants = sorted(
        set(r["participant"] for r in records)
    )

    print(
        "\n========== WITHIN-SUBJECT 10-FOLD CV (window-level) ==========\n"
    )
    print(
        f"{'Subject':>8}  {'Windows':>7}  "
        f"{'Win-Acc':>8}  {'Trial-Acc':>10}"
    )
    print("-" * 45)

    subject_win_accs   = []
    subject_trial_accs = []

    all_win_true  = []
    all_win_pred  = []
    all_trial_true = []
    all_trial_pred = []

    for participant in unique_participants:

        p_records = [
            r for r in records
            if r["participant"] == participant
        ]

        # Build window-level X and y, tracking trial membership
        X_all         = []
        y_win_all     = []    # window-level labels
        trial_idx_all = []    # which trial each window belongs to

        for t_idx, record in enumerate(p_records):

            n_win = len(record["features"])

            X_all.append(record["features"])
            y_win_all.extend([record["label"]] * n_win)
            trial_idx_all.extend([t_idx] * n_win)

        X_all         = np.vstack(X_all)
        y_win_all     = np.array(y_win_all)
        trial_idx_all = np.array(trial_idx_all)

        n_windows = len(y_win_all)

        # 10-fold stratified split on windows
        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        p_win_true  = []
        p_win_pred  = []
        p_trial_true = []
        p_trial_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X_all, y_win_all)
        ):

            X_tr, y_tr = X_all[train_idx], y_win_all[train_idx]
            X_te       = X_all[test_idx]
            y_te       = y_win_all[test_idx]
            trial_te   = trial_idx_all[test_idx]

            model = create_model(RANDOM_STATE + fold_idx)
            model.fit(X_tr, y_tr)

            win_preds = model.predict(X_te)

            # Window-level accumulation
            p_win_true.extend(y_te.tolist())
            p_win_pred.extend(win_preds.tolist())

            # Trial-level majority vote
            for t_idx in np.unique(trial_te):

                mask  = trial_te == t_idx
                voted = majority_vote(win_preds[mask])

                p_trial_true.append(p_records[t_idx]["label"])
                p_trial_pred.append(voted)

        win_acc   = accuracy_score(p_win_true,   p_win_pred)
        trial_acc = accuracy_score(p_trial_true, p_trial_pred)

        subject_win_accs.append(win_acc)
        subject_trial_accs.append(trial_acc)

        all_win_true.extend(p_win_true)
        all_win_pred.extend(p_win_pred)
        all_trial_true.extend(p_trial_true)
        all_trial_pred.extend(p_trial_pred)

        print(
            f"{participant:>8}  {n_windows:>7}  "
            f"{win_acc:>8.4f}  {trial_acc:>10.4f}"
        )

    # Overall evaluation
    print("\n========== FINAL RESULTS ==========")

    mean_win_acc   = np.mean(subject_win_accs)
    std_win_acc    = np.std(subject_win_accs)
    mean_trial_acc = np.mean(subject_trial_accs)
    std_trial_acc  = np.std(subject_trial_accs)

    overall_win_acc   = accuracy_score(all_win_true,   all_win_pred)
    overall_trial_acc = accuracy_score(all_trial_true, all_trial_pred)

    win_f1   = f1_score(all_win_true,   all_win_pred,   average="macro", zero_division=0)
    trial_f1 = f1_score(all_trial_true, all_trial_pred, average="macro", zero_division=0)

    print(
        f"Window-level  — Mean: {mean_win_acc:.4f} ±{std_win_acc:.4f}  "
        f"| Overall: {overall_win_acc:.4f} | F1: {win_f1:.4f}"
    )
    print(
        f"Trial-level   — Mean: {mean_trial_acc:.4f} ±{std_trial_acc:.4f}  "
        f"| Overall: {overall_trial_acc:.4f} | F1: {trial_f1:.4f}"
    )
    print(
        f"Number of subjects : {len(unique_participants)}"
    )

    print("\nWindow-level Classification Report:\n")

    print(
        classification_report(
            all_win_true,
            all_win_pred,
            labels=list(range(10)),
            zero_division=0
        )
    )

    print("Window-level Confusion Matrix:\n")

    print(
        confusion_matrix(
            all_win_true,
            all_win_pred,
            labels=list(range(10))
        )
    )

    return overall_win_acc


# ============================================================
# 10. TRAIN FINAL MODEL ON ALL RECORDINGS
# ============================================================

def train_final_model(records):

    X = np.vstack([
        r["features"]
        for r in records
    ])

    y = np.concatenate([

        np.full(
            len(r["features"]),
            r["label"],
            dtype=int
        )

        for r in records

    ])

    model = create_model()

    model.fit(X, y)

    artifact = {

        "model": model,

        "n_channels": EXPECTED_CHANNELS,

        "sampling_rate": EXPECTED_SAMPLING_RATE,

        "smoothing_window": SMOOTHING_WINDOW,

        "segment_duration": SEGMENT_DURATION,

        "n_features": EXPECTED_CHANNELS * 4,   # 56

        "classes": list(range(10))

    }

    joblib.dump(
        artifact,
        "eeg_digit_rf.joblib"
    )

    print(
        "\nFinal model saved as eeg_digit_rf.joblib"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("EEG IMAGINED DIGIT RECOGNITION")
    print("Kumar et al. 2017 — Envisioned Speech Recognition")
    print("Within-subject 10-fold CV — 50ms window, window-level accuracy")
    print("------------------------------------------------")

    # Load dataset
    records = load_dataset()

    # Evaluate using within-subject 10-fold CV (paper protocol)
    evaluate_model(records)

    # Train final model using all available data
    train_final_model(records)


if __name__ == "__main__":

    main()
