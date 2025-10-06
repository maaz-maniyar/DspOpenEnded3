import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# -----------------------
# 1. Load ECG data
# -----------------------
df = pd.read_excel('ECG.xlsx')

# Try common column names
for col in ['ECG','ecg','Signal','signal','Var1']:
    if col in df.columns:
        ecg_signal = df[col].dropna().values
        break
else:
    # fallback: first numeric column
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found in ECG.xlsx")
    ecg_signal = df[numeric_cols[0]].dropna().values

ecg_signal = ecg_signal.astype(float)
n = len(ecg_signal)

# Sampling frequency
fs = 360  # Hz
time = np.arange(n)/fs

# -----------------------
# 2. Bandpass filter (0.5-40 Hz)
# -----------------------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

ecg_bp = bandpass_filter(ecg_signal, fs)

# -----------------------
# 3. FFT - single sided magnitude spectrum
# -----------------------
dft = np.fft.fft(ecg_bp)
if n % 2 == 0:
    n_unique = n//2 + 1
    mag = np.abs(dft[:n_unique])/n
    mag[1:-1] *= 2
else:
    n_unique = (n+1)//2
    mag = np.abs(dft[:n_unique])/n
    mag[1:] *= 2
freqs = np.arange(n_unique)*(fs/n)

plt.figure()
plt.plot(freqs, mag, lw=1)
plt.xlim(0, min(60, freqs[-1]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Single-sided Magnitude Spectrum (0.5-40 Hz)')
plt.grid(True)

# -----------------------
# 4. R-peak detection (time-domain)
# -----------------------
min_peak_distance_s = 0.4  # minimum seconds between R-peaks
min_distance = int(min_peak_distance_s * fs)
prominence = 0.4 * np.std(ecg_bp)

peaks, _ = find_peaks(ecg_bp, distance=min_distance, prominence=prominence)

# If too few peaks, relax prominence
if len(peaks) < 2:
    peaks, _ = find_peaks(ecg_bp, distance=min_distance, prominence=0.2*np.std(ecg_bp))

# Time-domain HR
if len(peaks) < 2:
    heart_rate_time = np.nan
else:
    rr_intervals = np.diff(peaks)/fs
    heart_rate_time = 60 / np.mean(rr_intervals)

# -----------------------
# 5. Frequency-domain HR (physiological band 0.5-3.5 Hz)
# -----------------------
hr_band = (0.5, 3.5)
band_idx = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
if np.any(band_idx):
    dominant_freq = freqs[band_idx][np.argmax(mag[band_idx])]
    heart_rate_freq = dominant_freq * 60
else:
    heart_rate_freq = np.nan

print(f'Heart Rate (Time Domain): {heart_rate_time:.2f} BPM')
print(f'Heart Rate (Frequency Domain): {heart_rate_freq:.2f} BPM')

# -----------------------
# 6. Plot ECG with detected R-peaks
# -----------------------
plt.figure()
plt.plot(time, ecg_bp, 'b', label='Filtered ECG (0.5-40 Hz)')
plt.plot(peaks/fs, ecg_bp[peaks], 'ro', label='R-peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered ECG with Detected R-peaks')
plt.grid(True)
plt.legend()

# -----------------------
# 7. Moving average smoothing
# -----------------------
orders = [3,5]
plt.figure()
plt.plot(time, ecg_bp, color='0.6', label='Filtered ECG (BP)')
for order in orders:
    smoothed = np.convolve(ecg_bp, np.ones(order)/order, mode='same')
    plt.plot(time, smoothed, label=f'MA order {order}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Moving Average Smoothing')
plt.legend()
plt.grid(True)

plt.show()
