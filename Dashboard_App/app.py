import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import os

#1. APP CONFIGURATION
st.set_page_config(page_title="Direct Forecast Dashboard", page_icon="⚡", layout="wide")

#2. CONFIG & MODEL DEFINITIONS

MODEL_CONFIG = {
    "window_size": 50,
    "future_steps": 1200,
    "hidden_size": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "target_columns": ["degradation", "temperature_estimated", "time_since_last_maintenance"],
    "degradation_threshold": 0.66
}
MODEL_CONFIG["output_size"] = len(MODEL_CONFIG["target_columns"]) * MODEL_CONFIG["future_steps"]

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

#3. CACHED HELPER FUNCTIONS

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path, model_class, config):
    """Loads both the model and the scaler."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.sidebar.error("Model or scaler file not found. Check paths.")
        return None, None
    
    scaler = joblib.load(scaler_path)
    model_constructor_config = {
        "input_size": len(scaler.feature_names_in_),
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "output_size": config["output_size"],
        "dropout": config["dropout"]
    }
    model = model_class(**model_constructor_config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded file."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

#4. CORE FORECASTING FUNCTION

def run_direct_forecast(model, scaler, history_window_df, config):
    """
    Takes a history window, runs the direct forecast model, and returns the unscaled future predictions.
    """
    # Prepare the input window
    history_numeric = history_window_df.select_dtypes(include=[np.number])
    history_aligned = history_numeric.reindex(columns=scaler.feature_names_in_, fill_value=0)
    history_scaled = scaler.transform(history_aligned)
    input_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)

    # Run the model to get one flat prediction vector
    with torch.no_grad():
        flat_prediction_scaled = model(input_tensor)

    # Reshape the flat vector back to (future_steps, num_targets)
    num_targets = len(config["target_columns"])
    prediction_scaled = flat_prediction_scaled.cpu().numpy().reshape(config["future_steps"], num_targets)

    prediction_full_features = np.zeros((config["future_steps"], len(scaler.feature_names_in_)))
    target_indices = [list(scaler.feature_names_in_).index(c) for c in config["target_columns"]]
    
    prediction_full_features[:, target_indices] = prediction_scaled

    prediction_unscaled = scaler.inverse_transform(prediction_full_features)
    
    # Extract just the target columns we care about
    forecast_df = pd.DataFrame(prediction_unscaled[:, target_indices], columns=config["target_columns"])
    return forecast_df


#5. STREAMLIT UI

st.sidebar.title("Dashboard Controls")
page = st.sidebar.selectbox("Choose a Page", ["Live RUL Prediction", "Component Health Forecasting"])

st.title(f"⚡ {page}")

model, scaler = load_model_and_scaler("Dashboard_App/model_BiLSTM.pth", "Dashboard_App/main_scaler.joblib", BiLSTMModel, MODEL_CONFIG)

if model and scaler:
    # --- Sidebar Controls ---
    st.sidebar.header("Forecast Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Full History CSV", type="csv")
    
    if uploaded_file:
        df_history = load_data(uploaded_file)
        if df_history is not None and len(df_history) > MODEL_CONFIG["window_size"]:
            
            max_point = len(df_history) - 1
            min_point = MODEL_CONFIG["window_size"]
            
           
            prediction_point = st.sidebar.slider(
                "Select Forecast Starting Point", 
                min_value=min_point, 
                max_value=max_point, 
                value=max_point,
                help=f"The model will use the {MODEL_CONFIG['window_size']} steps before this point to forecast the future."
            )

            # --- Main Page Logic ---
            if page == "Live RUL Prediction":
                st.markdown("Derive the **Remaining Useful Life (RUL)** by forecasting the degradation curve until it hits the failure threshold.")
                if st.sidebar.button("Calculate RUL", key="rul_button"):
                    with st.spinner(f"Forecasting from step {prediction_point} to derive RUL..."):
                        
                        start_idx = prediction_point - MODEL_CONFIG["window_size"]
                        history_window_df = df_history.iloc[start_idx:prediction_point]

                        forecast_df = run_direct_forecast(model, scaler, history_window_df, MODEL_CONFIG)

                        forecasted_degradation = forecast_df['degradation'].values
                        try:
                            
                            predicted_rul = np.where(forecasted_degradation >= MODEL_CONFIG["degradation_threshold"])[0][0]
                        except IndexError:
                            
                            predicted_rul = -1 
                        
                        st.success("Analysis Complete!")
                        if predicted_rul != -1:
                            st.metric(label=f"Derived RUL from step {prediction_point}", value=f"{predicted_rul} Cycles")
                        else:
                            st.info(f"Degradation threshold was not reached within the {MODEL_CONFIG['future_steps']}-step forecast window. RUL is greater than {MODEL_CONFIG['future_steps']}.")

                        # Plotting
                        st.subheader("Degradation Forecast and RUL Point")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_history.index, y=df_history['degradation'], name='Historical Degradation', line=dict(color='royalblue')))
                        
                        forecast_index = np.arange(prediction_point, prediction_point + len(forecast_df))
                        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_df['degradation'], name='Forecasted Degradation', line=dict(color='red', dash='dash')))

                        if predicted_rul != -1:
                            failure_point_x = prediction_point + predicted_rul
                            failure_point_y = forecast_df['degradation'].iloc[predicted_rul]
                            fig.add_trace(go.Scatter(x=[failure_point_x], y=[failure_point_y], mode='markers', marker=dict(color='purple', size=15, symbol='x'), name=f'Predicted Failure (RUL={predicted_rul})'))
                        
                        fig.add_hline(y=MODEL_CONFIG["degradation_threshold"], line_dash="dot", line_color="orange", annotation_text="Failure Threshold")
                        fig.add_vline(x=prediction_point, line_dash="dot", line_color="green", annotation_text="Forecast Start")
                        
                        fig.update_layout(title="<b>Degradation History and Forecast</b>", xaxis_title="Time Step", yaxis_title="Degradation")
                        st.plotly_chart(fig, use_container_width=True)


            elif page == "Component Health Forecasting":
                st.markdown("Forecast the future trajectory of all target components in a single prediction.")
                targets_to_plot = st.sidebar.multiselect("Select Targets to Plot", options=MODEL_CONFIG["target_columns"], default=MODEL_CONFIG["target_columns"])

                if st.sidebar.button("Forecast Components", key="health_button"):
                    with st.spinner(f"Forecasting all components from step {prediction_point}..."):
                        
                        start_idx = prediction_point - MODEL_CONFIG["window_size"]
                        history_window_df = df_history.iloc[start_idx:prediction_point]
                        
                        forecast_df = run_direct_forecast(model, scaler, history_window_df, MODEL_CONFIG)
                        
                        st.success("Forecast Complete!")
                        st.write("Forecasted Data Preview:", forecast_df.head())

                        for target in targets_to_plot:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_history.index, y=df_history[target], name=f'Historical {target}', line=dict(color='royalblue')))
                            
                            forecast_index = np.arange(prediction_point, prediction_point + len(forecast_df))
                            fig.add_trace(go.Scatter(x=forecast_index, y=forecast_df[target], name=f'Forecasted {target}', line=dict(color='red', dash='dash')))
                            
                            fig.add_vline(x=prediction_point, line_dash="dot", line_color="green", annotation_text="Forecast Start")
                            
                            fig.update_layout(title=f"<b>{target.replace('_', ' ').title()}: History and Direct Forecast</b>", xaxis_title="Time Step", yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a data file using the sidebar to begin.")
else:
    st.error("Could not load the model and/or scaler. Please check the file paths and ensure they are in the root directory with the app.")