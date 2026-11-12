"""
Generate publication-quality figures for the Enhanced EEG Imagined Digit Classifier.
Saves high-res (300 DPI) PNG and vector PDF files suitable for IEEE/Elsevier/Nature papers.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Set publication style defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Create output folder
OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Load metrics
with open("enhanced_eval_metrics.json", "r") as f:
    data = json.load(f)

cm = np.array(data["confusion_matrix"])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

per_class = data["per_class"]
digits = [str(x["digit"]) for x in per_class]
precisions = [x["precision"] * 100 for x in per_class]
recalls = [x["recall"] * 100 for x in per_class]
f1s = [x["f1"] * 100 for x in per_class]

sub_details = data["subject_details"]
sub_ids = [str(s["participant"]) for s in sub_details]
sub_win_accs = [s["win_acc"] * 100 for s in sub_details]
sub_trial_accs = [s["trial_acc"] * 100 for s in sub_details]

n_parts = data["n_participants"]
n_recs = data["n_recordings"]
n_wins = data["n_windows"]
feat_dim = data["feature_dim"]

win_mean = data["mean_participant_win_acc"] * 100
win_std = data["std_participant_win_acc"] * 100
win_pooled = data["overall_win_acc"] * 100
win_macro_f1 = data["overall_win_macro_f1"] * 100

trial_mean = data["mean_participant_trial_acc"] * 100
trial_std = data["std_participant_trial_acc"] * 100
trial_pooled = data["overall_trial_acc"] * 100
trial_macro_f1 = data["overall_trial_macro_f1"] * 100


# ==============================================================================
# FIGURE 1: CONFUSION MATRIX
# ==============================================================================
def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=18, fontweight='semibold')
    cbar.ax.tick_params(labelsize=10)
    
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([f"Digit {i}" for i in range(10)], fontweight='semibold')
    ax.set_yticklabels([f"Digit {i}" for i in range(10)], fontweight='semibold')
    
    # Text annotations in each cell
    thresh = cm_norm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val_pct = cm_norm[i, j]
            val_cnt = cm[i, j]
            color = "white" if val_pct > thresh else "black"
            if val_pct >= 1.0 or i == j:
                text = f"{val_pct:.1f}%\n({val_cnt})"
            else:
                text = f"{val_pct:.1f}%"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color=color,
                    fontsize=8.5 if i == j else 7.5,
                    fontweight='bold' if i == j else 'normal')
            
    ax.set_ylabel('True Digit Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel('Predicted Digit Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f"Enhanced Model Confusion Matrix (10-Class Digit Recognition)\n"
        f"Pooled Window-Level Accuracy: {win_pooled:.2f}% | Total Windows: {n_wins:,} across {n_parts} Subjects",
        fontsize=12.5, fontweight='bold', pad=14
    )
    
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig1_confusion_matrix.png")
    pdf_path = os.path.join(OUT_DIR, "fig1_confusion_matrix.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# FIGURE 2: PER-CLASS PRECISION, RECALL, F1-SCORE
# ==============================================================================
def plot_per_class_metrics():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = np.arange(10)
    width = 0.26
    
    c_p = "#1f77b4"  # Deep blue
    c_r = "#2ca02c"  # Fresh green
    c_f = "#ff7f0e"  # Amber orange
    
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color=c_p, edgecolor='black', linewidth=0.6, alpha=0.92)
    rects2 = ax.bar(x, recalls, width, label='Recall', color=c_r, edgecolor='black', linewidth=0.6, alpha=0.92)
    rects3 = ax.bar(x + width, f1s, width, label='F1-Score', color=c_f, edgecolor='black', linewidth=0.6, alpha=0.92)
    
    # Baseline benchmark line
    paper_baseline = 68.5
    ax.axhline(paper_baseline, color='#d62728', linestyle='--', linewidth=1.4, alpha=0.85,
               label=f'Kumar et al. (2017) Baseline ({paper_baseline}%)')
    ax.axhline(win_pooled, color='#2b5c8f', linestyle=':', linewidth=1.5, alpha=0.85,
               label=f'Enhanced Pooled Acc ({win_pooled:.1f}%)')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Imagined Digit Class', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Per-Class Performance Metrics (Precision, Recall, F1-Score)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Digit {i}" for i in range(10)], fontweight='semibold')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Value annotations on F1 bars
    for i, rect in enumerate(rects3):
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc', fontsize=9.5)
    
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig2_per_class_metrics.png")
    pdf_path = os.path.join(OUT_DIR, "fig2_per_class_metrics.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# FIGURE 3: PARTICIPANT-WISE WINDOW ACCURACY
# ==============================================================================
def plot_participant_breakdown():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    x = np.arange(len(sub_ids))
    width = 0.62
    
    bars = ax.bar(x, sub_win_accs, width, label='Window-Level Accuracy',
                  color='#3b82f6', edgecolor='#1d4ed8', linewidth=0.8, alpha=0.92)
    
    # Paper baseline reference line
    ax.axhline(68.5, color='#d62728', linestyle='--', linewidth=1.4, alpha=0.85,
               label='Kumar et al. (2017) Baseline (68.50%)')
    
    # Mean line and shaded std band for window-level
    ax.axhline(win_mean, color='#1e3a8a', linestyle='-', linewidth=1.6,
               label=f'Mean Window Acc ({win_mean:.2f}%)')
    ax.axhspan(win_mean - win_std, win_mean + win_std, color='#bfdbfe', alpha=0.30,
               label=f'Window Acc ±1 SD (±{win_std:.2f}%)')
    
    ax.set_ylabel('Window Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Participant (Subject ID)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(f'Participant-Wise Window-Level Accuracy (23 Subjects)\n'
                 f'Mean: {win_mean:.2f}% ± {win_std:.2f}% | Pooled: {win_pooled:.2f}%',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{sid}" for sid in sub_ids], rotation=0, fontsize=9.5, fontweight='semibold')
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, fontweight='bold', rotation=90)
        
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc', ncol=3, fontsize=9)
    
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig3_participant_performance.png")
    pdf_path = os.path.join(OUT_DIR, "fig3_participant_performance.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# FIGURE 4: COMPREHENSIVE EXPERIMENTAL SUMMARY CARD / TABLE
# ==============================================================================
def plot_summary_table():
    fig = plt.figure(figsize=(10.5, 6.2))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table_data = [
        ["Evaluation Metric / Attribute", "Enhanced Approach Value", "Benchmark / Context"],
        ["Participants Evaluated", f"{n_parts} Subjects (Subject 0 to 24)", "All available subjects with full 14 channels"],
        ["Total EEG Recordings (Trials)", f"{n_recs} EDF recordings", "10 digit trials (0-9) per subject"],
        ["Total Evaluated Windows", f"{n_wins:,} windows", "50 ms duration with 50% overlap (25 ms hop)"],
        ["Feature Vector Dimensionality", f"{feat_dim}-D features per window", "56 Time-Domain + 70 Band-Power + 42 Hjorth + 14 ZCR"],
        ["Window-Level Accuracy (Mean ± SD)", f"{win_mean:.2f}% ± {win_std:.2f}%", "Across 23 participants (10-fold positional CV)"],
        ["Window-Level Pooled Accuracy", f"{win_pooled:.2f}%", f"+14.84% gain over Kumar et al. (68.50%)"],
        ["Window-Level Macro-F1", f"{win_macro_f1:.2f}%", "Balanced across all 10 digit classes (0-9)"],
        ["Corrected Trial-Level Accuracy (Mean ± SD)", f"{trial_mean:.2f}% ± {trial_std:.2f}%", "Majority voting over constituent trial windows"],
        ["Corrected Trial-Level Pooled Accuracy", f"{trial_pooled:.2f}%", f"229/230 trials correctly classified"],
        ["Corrected Trial-Level Macro-F1", f"{trial_macro_f1:.2f}%", "Near-perfect trial-level discrimination"],
        ["Cross-Validation Strategy", "Within-subject 10-fold positional CV", "Zero temporal leakage (stride-modulo split)"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.42, 0.28, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.65)
    
    # Style header row
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#1e3a8a')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)
        cell.set_height(0.08)
        
    # Style alternating rows
    for i in range(1, len(table_data)):
        row_color = '#f8fafc' if i % 2 == 0 else '#ffffff'
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor(row_color)
            if j == 1:
                cell.set_text_props(fontweight='bold', color='#0f172a')
            if j == 0:
                cell.set_text_props(fontweight='semibold', color='#1e293b')
            cell.set_edgecolor('#cbd5e1')
            cell.set_linewidth(0.7)
            
    plt.title("Enhanced EEG Imagined Digit Recognition System — Experimental Summary",
              fontsize=13, fontweight='bold', pad=25)
    
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig4_summary_table.png")
    pdf_path = os.path.join(OUT_DIR, "fig4_summary_table.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


# ==============================================================================
# FIGURE 5: COMPOSITE MULTI-PANEL JOURNAL FIGURE (2x2)
# ==============================================================================
def plot_composite_panel():
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
    
    # ----------------- PANEL A: Confusion Matrix -----------------
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=15, fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    
    tick_marks = np.arange(10)
    ax_a.set_xticks(tick_marks)
    ax_a.set_yticks(tick_marks)
    ax_a.set_xticklabels([f"D{i}" for i in range(10)], fontsize=8.5)
    ax_a.set_yticklabels([f"D{i}" for i in range(10)], fontsize=8.5)
    
    thresh = cm_norm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val_pct = cm_norm[i, j]
            color = "white" if val_pct > thresh else "black"
            ax_a.text(j, i, f"{val_pct:.1f}%",
                      ha="center", va="center", color=color,
                      fontsize=7.2, fontweight='bold' if i == j else 'normal')
            
    ax_a.set_ylabel('True Digit Label', fontsize=10.5, fontweight='bold')
    ax_a.set_xlabel('Predicted Digit Label', fontsize=10.5, fontweight='bold')
    ax_a.set_title('(A) 10-Class Confusion Matrix (Pooled Acc: 83.34%)',
                   fontsize=11.5, fontweight='bold', loc='left')
    
    # ----------------- PANEL B: Per-Class Metrics -----------------
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(10)
    w = 0.26
    ax_b.bar(x - w, precisions, w, label='Precision', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax_b.bar(x, recalls, w, label='Recall', color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax_b.bar(x + w, f1s, w, label='F1-Score', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    ax_b.axhline(68.5, color='#d62728', linestyle='--', linewidth=1.2, label='Paper Baseline (68.5%)')
    ax_b.set_ylabel('Score (%)', fontsize=10.5, fontweight='bold')
    ax_b.set_xlabel('Imagined Digit Class', fontsize=10.5, fontweight='bold')
    ax_b.set_title('(B) Per-Class Precision, Recall, and F1-Score',
                   fontsize=11.5, fontweight='bold', loc='left')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"D{i}" for i in range(10)], fontsize=8.5)
    ax_b.set_ylim(0, 105)
    ax_b.grid(axis='y', linestyle='--', alpha=0.35)
    ax_b.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    
    # ----------------- PANEL C: Participant Breakdown -----------------
    ax_c = fig.add_subplot(gs[1, 0])
    xp = np.arange(len(sub_ids))
    wp = 0.62
    ax_c.bar(xp, sub_win_accs, wp, label='Window-Level Acc', color='#3b82f6', edgecolor='#1d4ed8', linewidth=0.6, alpha=0.92)
    ax_c.axhline(68.5, color='#d62728', linestyle='--', linewidth=1.1, label='Baseline (68.5%)')
    ax_c.axhline(win_mean, color='#1e3a8a', linestyle='-', linewidth=1.3, label=f'Mean ({win_mean:.1f}%)')
    ax_c.axhspan(win_mean - win_std, win_mean + win_std, color='#bfdbfe', alpha=0.25, label=f'±1 SD (±{win_std:.1f}%)')
    
    ax_c.set_ylabel('Window Accuracy (%)', fontsize=10.5, fontweight='bold')
    ax_c.set_xlabel('Participant (Subject ID)', fontsize=10.5, fontweight='bold')
    ax_c.set_title('(C) Participant-Wise Window-Level Accuracy (23 Subjects)',
                   fontsize=11.5, fontweight='bold', loc='left')
    ax_c.set_xticks(xp)
    ax_c.set_xticklabels([f"S{sid}" for sid in sub_ids], rotation=90, fontsize=8)
    ax_c.set_ylim(0, 108)
    ax_c.grid(axis='y', linestyle='--', alpha=0.35)
    ax_c.legend(loc='lower right', fontsize=8, ncol=2, framealpha=0.9)
    
    # ----------------- PANEL D: Overall Comparison & Key Metrics -----------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    
    summary_text = [
        ["Key Evaluation Parameter", "Enhanced Value", "Reference / Notes"],
        ["Evaluated Participants", f"{n_parts} Subjects", "Subject 0 to 24"],
        ["Total EEG Recordings (Trials)", f"{n_recs} Files", "10 digits per subject"],
        ["Total Analyzed Windows", f"{n_wins:,} Windows", "50ms window, 25ms hop"],
        ["Feature Vector Size", f"{feat_dim}-D Features", "Time + Welch + Hjorth + ZCR"],
        ["Window Accuracy (Mean ± SD)", f"{win_mean:.2f}% ± {win_std:.2f}%", f"Range: {min(sub_win_accs):.1f}% – {max(sub_win_accs):.1f}%"],
        ["Window Pooled Accuracy", f"{win_pooled:.2f}%", "Δ = +14.84% over paper"],
        ["Window Macro-F1 Score", f"{win_macro_f1:.2f}%", "10-class macro average"],
        ["Trial Accuracy (Mean ± SD)", f"{trial_mean:.2f}% ± {trial_std:.2f}%", "Majority voting rule"],
        ["Trial Pooled Accuracy", f"{trial_pooled:.2f}%", "229 of 230 trials correct"],
        ["Trial Macro-F1 Score", f"{trial_macro_f1:.2f}%", "Across all 10 classes"]
    ]
    
    t_d = ax_d.table(cellText=summary_text, cellLoc='left', loc='center', colWidths=[0.42, 0.28, 0.30])
    t_d.auto_set_font_size(False)
    t_d.set_fontsize(8.8)
    t_d.scale(1.0, 1.7)
    
    for j in range(3):
        t_d[0, j].set_facecolor('#1e3a8a')
        t_d[0, j].set_text_props(color='white', fontweight='bold', fontsize=10.0)
        
    for i in range(1, len(summary_text)):
        row_c = '#f1f5f9' if i % 2 == 0 else '#ffffff'
        for j in range(3):
            cell = t_d[i, j]
            cell.set_facecolor(row_c)
            cell.set_edgecolor('#cbd5e1')
            if j == 1:
                cell.set_text_props(fontweight='bold', color='#0f172a')
            if j == 0:
                cell.set_text_props(fontweight='semibold', color='#1e293b')
                
    ax_d.set_title('(D) Experimental Benchmark Summary', fontsize=11.5, fontweight='bold', loc='left', pad=14)
    
    plt.suptitle("Enhanced EEG Imagined Digit Recognition — Comprehensive Experimental Evaluation",
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "fig5_composite_paper_panel.png")
    pdf_path = os.path.join(OUT_DIR, "fig5_composite_paper_panel.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_per_class_metrics()
    plot_participant_breakdown()
    plot_summary_table()
    plot_composite_panel()
    print(f"\nAll publication figures successfully generated in '{OUT_DIR}' directory!")
