
import mne

filepath = "./data/Digit/name1_0.edf"

raw = mne.io.read_raw_edf(
    filepath,
    preload=True,
    verbose="ERROR"
)

print("Total channels:", len(raw.ch_names))
print("Sampling frequency:", raw.info["sfreq"])
print("Duration:", raw.n_times / raw.info["sfreq"])

print("\nChannel names:")

for i, name in enumerate(raw.ch_names):
    print(i, name)

print("\nChannel types:")

for name, channel_type in zip(
    raw.ch_names,
    raw.get_channel_types()
):
    print(name, ":", channel_type)
