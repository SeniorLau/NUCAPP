import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Nucleation Aligner", layout="centered")

st.title("🧪 Freezing curve aligner")

# --- Tabs ---
tab_main, tab_settings = st.tabs(["Main", "Settings"])

with tab_settings:
    st.subheader("🔧 Settings")
    ci_level = st.slider("Confidence Interval Level (%)", min_value=80, max_value=99, value=95, step=1)

# Default confidence if not yet set
if 'ci_level' not in locals():
    ci_level = 95

alpha = 1 - (ci_level / 100)

with tab_main:
    # --- File Upload ---
    uploaded_file = st.file_uploader("Upload your semicolon-separated CSV (comma decimal):", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',')
            st.success("✅ File loaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
    else:
        st.info("Upload a CSV file to begin.")
        st.stop()

    # User Inputs
    n_before = st.number_input("Points BEFORE nucleation:", min_value=0, max_value=1000, value=50)
    n_after = st.number_input("Points AFTER nucleation:", min_value=0, max_value=2000, value=320)
    XLIM = st.number_input("Xlim of graph:", min_value=0, max_value=1000, value=50)

    # Nucleation points
    st.subheader("Step 1: Click nucleation points on plot")
    time_min = np.array(df['Name']) / 1000 / 60
    temperature = np.array(df['fTemperatureCorrected'])

    fig1, ax1 = plt.subplots()
    ax1.plot(time_min, temperature)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title("Nucleation points")
    ax1.grid(True)
    st.pyplot(fig1)

    st.warning("Streamlit does not support interactive point selection via matplotlib in browser.")
    manual_points = st.text_input("Enter nucleation points (comma-separated, in minutes):", "2.5, 4.0, 5.8")
    try:
        selected_points = [(float(x.strip()), 0) for x in manual_points.split(",") if x.strip()]
    except:
        st.error("Invalid input format.")
        st.stop()

    if len(selected_points) == 0:
        st.warning("You must enter at least one nucleation point.")
        st.stop()

    n = len(selected_points)
    t_value = stats.t.ppf(1 - alpha / 2, df=n - 1) if n < 30 else stats.norm.ppf(1 - alpha / 2)

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
        sem_curve = std_curve / np.sqrt(n)
        return mean_curve, sem_curve, aligned_segments

    def ci(mean, sem): return (mean - t_value * sem, mean + t_value * sem)

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
        results[name] = {'mean': mean, 'sem': sem, 'curves': curves, 'ci': ci(mean, sem)}

    # Plotting temperature
    st.subheader("Aligned Temperature Curves")
    fig_temp, ax = plt.subplots()
    colors = {'Corrected Temp': 'blue', 'Cylinder Temp': 'green', 'Gas Temp': 'red'}
    for label in ['Corrected Temp', 'Cylinder Temp', 'Gas Temp']:
        for curve in results[label]['curves']:
            ax.plot(curve, color=colors[label], alpha=0.3, linestyle='--')
        ax.plot(results[label]['mean'], label=f"{label} (mean)", color=colors[label], linewidth=2)
        ax.fill_between(range(len(results[label]['mean'])), *results[label]['ci'], color=colors[label], alpha=0.15)
    ax.axvline(n_before, color='black', linestyle='--', label='Nucleation Point')
    ax.set_xlim([0, XLIM])
    ax.set_ylim([-90, 20])
    ax.grid(True)
    ax.legend()
    st.pyplot(fig_temp)

    # Plotting flows
    st.subheader("Inlet & Extraction Flows")
    fig_flow, ax = plt.subplots()
    colors = {'Inlet Flow': 'purple', 'Extraction Flow': 'orange'}
    for label in ['Inlet Flow', 'Extraction Flow']:
        for curve in results[label]['curves']:
            ax.plot(curve, color=colors[label], alpha=0.3, linestyle='--')
        ax.plot(results[label]['mean'], label=f"{label} (mean)", color=colors[label], linewidth=2)
        ax.fill_between(range(len(results[label]['mean'])), *results[label]['ci'], color=colors[label], alpha=0.15)
    ax.axvline(n_before, color='black', linestyle='--', label='Nucleation Point')
    ax.grid(True)
    ax.set_xlim([0, XLIM])
    ax.legend()
    st.pyplot(fig_flow)
