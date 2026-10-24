import os
import glob
import warnings
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress minor convergence warnings for clean terminal output
warnings.filterwarnings('ignore')

def load_and_clean_emotiv_dataset(data_dir, epoch_duration=1.0):
    edf_files = glob.glob(os.path.join(data_dir, "*.edf"))
    print(f"Found {len(edf_files)} EDF files in {data_dir}")
    
    all_epochs = []
    all_labels = []
    
    eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    for file_path in edf_files:
        base_name = os.path.basename(file_path)
        try:
            label_str = base_name.split('_')[-1].split('.')[0]
            label = int(label_str)
        except ValueError:
            continue
            
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        available_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
        if not available_chs:
            available_chs = raw.ch_names[:14]
            
        raw.pick(available_chs)
        
        # 1. High-pass filter at 1.0 Hz for stable ICA decomposition
        raw_ica_compat = raw.copy().filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
        
        # 2. Fit ICA and remove high-kurtosis artifact components (eye blinks, muscle noise)
        try:
            ica = ICA(n_components=min(10, len(available_chs)-1), method='fastica', random_state=42, max_iter=1000, tol=1e-3)
            ica.fit(raw_ica_compat, verbose=False)
            
            reject_idx = []
            for idx in range(ica.n_components_):
                sources = ica.get_sources(raw_ica_compat).get_data()[idx]
                if kurtosis(sources) > 5.0:
                    reject_idx.append(idx)
            
            ica.exclude = reject_idx[:2]
            ica.apply(raw, verbose=False)
        except Exception:
            pass  # Fallback gracefully if edge-case files don't converge
            
        # Standard bandpass filter for feature extraction (0.5 - 40 Hz)
        raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
        
        events = mne.make_fixed_length_events(raw, duration=epoch_duration, overlap=0.5)
        if len(events) == 0:
            continue
            
        epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=epoch_duration, 
                            baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data()
        for epoch in data:
            all_epochs.append(epoch)
            all_labels.append(label)
            
    return np.array(all_epochs), np.array(all_labels)

def extract_features(X, sfreq=128):
    n_trials, n_channels, n_samples = X.shape
    feature_list = []

    for i in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            sig = X[i, ch, :]
            
            # Time-Domain Statistics
            mean_val = np.nan_to_num(np.mean(sig))
            std_val = np.nan_to_num(np.std(sig))
            var_val = np.nan_to_num(np.var(sig))
            rms_val = np.nan_to_num(np.sqrt(np.mean(sig**2)))
            p2p_val = np.nan_to_num(np.ptp(sig))
            skew_val = np.nan_to_num(skew(sig))
            kurt_val = np.nan_to_num(kurtosis(sig))
            line_length = np.nan_to_num(np.sum(np.abs(np.diff(sig))))
            zero_crossings = np.nan_to_num(np.where(np.diff(np.signbit(sig)))[0].size / n_samples)

            time_feats = [
                mean_val, std_val, var_val, rms_val, p2p_val, 
                skew_val, kurt_val, line_length, zero_crossings
            ]
            
            # Frequency-Domain PSD
            fft_vals = np.fft.rfft(sig)
            fft_freqs = np.fft.rfftfreq(n_samples, d=1.0/sfreq)
            fft_power = np.abs(fft_vals)**2
            
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
            
            band_powers = []
            for band, (low, high) in bands.items():
                idx_band = np.where((fft_freqs >= low) & (fft_freqs <= high))
                power = np.mean(fft_power[idx_band]) if len(idx_band[0]) > 0 else 0.0
                band_powers.append(np.nan_to_num(power))

            trial_features.extend(time_feats + band_powers)
            
        feature_list.append(trial_features)

    features_array = np.array(feature_list)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    return features_array

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\ASUS\OneDrive\Desktop\EEG_Project\Digit"
    
    print("Loading EDF files with ICA Artifact Removal...")
    X_raw, y_raw = load_and_clean_emotiv_dataset(DATA_DIR, epoch_duration=1.0)
    
    print(f"Loaded dataset shape - X: {X_raw.shape}, y: {y_raw.shape}")
    print(f"Unique labels found: {np.unique(y_raw)}")
    
    if len(X_raw) > 0:
        print("Extracting features from ICA-cleaned signals...")
        X_features = extract_features(X_raw)
        print(f"Feature matrix shape: {X_features.shape}")
        
        print("Training Ensemble Classifier with 5-Fold Cross-Validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y_raw)):
            X_train, X_test = X_features[train_idx], X_features[test_idx]
            y_train, y_test = y_raw[train_idx], y_raw[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            rf_clf = RandomForestClassifier(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1)
            et_clf = ExtraTreesClassifier(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1)
            
            ensemble_clf = VotingClassifier(
                estimators=[('rf', rf_clf), ('et', et_clf)],
                voting='soft'
            )
            
            ensemble_clf.fit(X_train, y_train)
            
            y_pred = ensemble_clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            print(f"Fold {fold+1} Accuracy: {acc * 100:.2f}%")
            
        print(f"\nMean ICA-Cleaned Ensemble Accuracy: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")
        
        print("\n--- Detailed Classification Report Across All Digits (0-9) ---")
        print(classification_report(all_y_true, all_y_pred, digits=4))