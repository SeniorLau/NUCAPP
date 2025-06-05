import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Nucleation Aligner", layout="centered")

st.title("🧪 Freezing curve aligner")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your semicolon-separated CSV (comma decimal):", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        st.success("✅ File loaded successfully!")
        st.write("First few rows of data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()
else:
    st.info("Upload a CSV file to begin.")
    st.stop()

# --- User Inputs ---
n_before = st.slider("Points BEFORE nucleation:", min_value=0, max_value=500, value=50, step=5)
n_after = st.slider("Points AFTER nucleation:", min_value=0, max_value=2000, value=320, step=10)
XLIM = st.number_input("Xlim of graph:", min_value=0, max_value=1000, value=50)
# --- Plot and select nucleation points ---
st.subheader("Step 1: Click nucleation points on plot")

time_min = np.array(df['Name']) / 1000 / 60
temperature = np.array(df['fTemperatureCorrected'])

fig1, ax1 = plt.subplots()
ax1.plot(time_min, temperature, label='Corrected Temperature')
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title("Click nucleation points (at least 1)")
ax1.grid(True)

selected = st.pyplot(fig1, use_container_width=True)
st.write("⚠️ Use the Matplotlib window (if interactive) to select points. Streamlit does not support `plt.ginput()` directly.")

st.warning("👉 Because Streamlit is browser-based, **interactive point selection is not supported natively.**")
st.markdown("""
- **Workaround:** Manually enter the x-values (time in minutes) where nucleation occurs.
""")

# --- Step 1: Ask how many nucleation points to select ---
st.subheader("Step 1: Define nucleation points")

max_possible_points = 10  # Optional upper limit
num_points = st.slider("How many nucleation points do you want to define?", min_value=1, max_value=max_possible_points, value=3)

# --- Step 2: Create a slider per nucleation point ---
unique_times = sorted(set(np.round(time_min, 2)))
selected_points = []

st.markdown("### Select nucleation times (in minutes):")

for i in range(num_points):
    selected_time = st.select_slider(
        label=f"Nucleation point {i + 1}",
        options=unique_times,
        value=unique_times[min(i, len(unique_times) - 1)]
    )
    selected_points.append((selected_time, 0))  # We still need (x, 0) format




n = len(selected_points)
t_value = stats.t.ppf(0.975, df=n - 1) if n < 30 else 1.96

def get_aligned_stats(column_data):
    aligned_segments = []
    total_len = n_before + n_after

    for x_click, _ in selected_points:
        idx = (np.abs(time_min - x_click)).argmin()
        start = max(0, idx - n_before)
        end = min(len(column_data), idx + n_after)
        segment = column_data[start:end]

        if len(segment) < total_len:
            segment = np.pad(segment, (0, total_len - len(segment)), constant_values=np.nan)

        aligned_segments.append(segment)

    aligned_segments = np.array(aligned_segments)
    mean_curve = np.nanmean(aligned_segments, axis=0)
    std_curve = np.nanstd(aligned_segments, axis=0)
    sem_curve = std_curve
    return mean_curve, sem_curve, aligned_segments

def ci(mean, sem): return (mean - t_value * sem, mean + t_value * sem)

# Process each column
columns = {
    'Corrected Temp': 'fTemperatureCorrected',
    'Cylinder Temp': 'fCylinderTemperatureActual',
    'Gas Temp': 'fGasTemperatureActual',
    'Inlet Flow': 'fInlet',
    'Extraction Flow': 'fExtraction'
}

results = {}
for name, col in columns.items():
    mean, sem, curves = get_aligned_stats(np.array(df[col]))
    results[name] = {
        'mean': mean,
        'sem': sem,
        'curves': curves,
        'ci': ci(mean, sem)
    }

# --- Plot Temperature Curves ---
st.subheader("Aligned Temperature Curves")

fig_temp, ax = plt.subplots(figsize=(8, 5))
colors = {'Corrected Temp': 'blue', 'Cylinder Temp': 'green', 'Gas Temp': 'red'}

for label in ['Corrected Temp', 'Cylinder Temp', 'Gas Temp']:
    for curve in results[label]['curves']:
        ax.plot(curve, color=colors[label], alpha=0.3, linestyle='--')
    ax.plot(results[label]['mean'], label=f"{label} (mean)", color=colors[label], linewidth=2)
    ax.fill_between(range(len(results[label]['mean'])), *results[label]['ci'], color=colors[label], alpha=0.15)

ax.axvline(n_before, color='black', linestyle='--', label='Nucleation Point')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Curves with 95% Confidence Intervals')
ax.grid(True)
ax.set_ylim([-90, 20])
ax.set_xlim([0, XLIM])
ax.legend()
st.pyplot(fig_temp)

# --- Plot Flow Curves ---
st.subheader("Inlet & Extraction Flows")

fig_flow, ax = plt.subplots(figsize=(8, 5))
colors = {'Inlet Flow': 'purple', 'Extraction Flow': 'orange'}

for label in ['Inlet Flow', 'Extraction Flow']:
    for curve in results[label]['curves']:
        ax.plot(curve, color=colors[label], alpha=0.3, linestyle='--')
    ax.plot(results[label]['mean'], label=f"{label} (mean)", color=colors[label], linewidth=2)
    ax.fill_between(range(len(results[label]['mean'])), *results[label]['ci'], color=colors[label], alpha=0.15)

ax.axvline(n_before, color='black', linestyle='--', label='Nucleation Point')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow')
ax.set_title('Flow Curves with 95% Confidence Intervals')
ax.grid(True)
ax.legend()
st.pyplot(fig_flow)
