import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Stock Close Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }
    .sub-text {
        font-size: 17px;
        color: #4b5563;
        margin-bottom: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 16px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .metric-box {
        padding: 18px;
        border-radius: 14px;
        background: linear-gradient(135deg, #ecfeff, #eef2ff);
        border: 1px solid #dbeafe;
        text-align: center;
    }
    .metric-title {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #111827;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Constants
# ----------------------------
N_PAST = 14
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Change %", "Volume"]
TARGET_INDEX = 3  # Close

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("Stock_Model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts the **next Close price**
    using the last **14 rows** of stock data.

    **Required columns:**
    - Open
    - High
    - Low
    - Close
    - Change %
    - Volume
    """
)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="main-title">📈 Stock Close Price Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Upload a CSV file, explore the stock data, and predict the next closing price.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload stock CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        # Clean numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean Change %
        df["Change %"] = (
            df["Change %"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce")

        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

        if len(df) == 0:
            st.error("No valid rows remain after cleaning the data.")
            st.stop()

        # ----------------------------
        # Overview metrics
        # ----------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Available", len(df))
        with col2:
            st.metric("Latest Close", f"{df['Close'].iloc[-1]:.4f}")
        with col3:
            st.metric("Average Volume", f"{df['Volume'].mean():,.0f}")

        st.divider()

        # ----------------------------
        # Plot section
        # ----------------------------
        st.subheader("Visualization")

        left_col, right_col = st.columns([1, 2])

        with left_col:
            plot_col = st.selectbox("Select a column to plot", FEATURE_COLUMNS)
            plot_btn = st.button("Plot Selected Column", use_container_width=True)

        with right_col:
            if plot_btn:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df[plot_col].values, marker="o", linewidth=2)
                ax.set_title(f"{plot_col} Trend", fontsize=14, fontweight="bold")
                ax.set_xlabel("Index")
                ax.set_ylabel(plot_col)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        st.divider()

        # ----------------------------
        # Prediction section
        # ----------------------------
        st.subheader("Prediction")

        if len(df) < N_PAST:
            st.warning(f"Need at least {N_PAST} rows to make prediction.")
        else:
            if st.button("Predict Next Close Price", use_container_width=True):
                latest_data = df[FEATURE_COLUMNS].tail(N_PAST).astype(float).values

                # Scale
                scaled_data = scaler.transform(latest_data)

                # Reshape for LSTM
                x_input = np.reshape(scaled_data, (1, N_PAST, len(FEATURE_COLUMNS)))

                # Predict
                pred_scaled = model.predict(x_input, verbose=0)

                # Inverse transform
                dummy_array = np.zeros((1, len(FEATURE_COLUMNS)))
                dummy_array[0, TARGET_INDEX] = pred_scaled[0][0]
                predicted_close = scaler.inverse_transform(dummy_array)[0, TARGET_INDEX]

                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-title">Predicted Next Close Price</div>
                        <div class="metric-value">{predicted_close:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Forecast plot
                historical_close = df["Close"].tail(20).tolist()
                x_hist = list(range(1, len(historical_close) + 1))
                forecast_x = x_hist[-1] + 1

                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(
                    x_hist,
                    historical_close,
                    marker="o",
                    linewidth=2,
                    label="Historical Close"
                )
                ax.plot(
                    [forecast_x],
                    [predicted_close],
                    marker="o",
                    markersize=10,
                    label="Forecasted Close"
                )
                ax.plot(
                    [x_hist[-1], forecast_x],
                    [historical_close[-1], predicted_close],
                    linestyle="--",
                    linewidth=2
                )
                ax.set_title("Close Price Forecast", fontsize=14, fontweight="bold")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Close Price")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                st.subheader("Last 14 Rows Used for Prediction")
                st.dataframe(df[FEATURE_COLUMNS].tail(N_PAST), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file to continue.")
