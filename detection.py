import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="EEG Seizure Detection System",
    page_icon="🧠",  # Using emoji instead of file path for cloud deployment
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load logo safely
def load_logo():
    """Load logo with fallback for cloud deployment"""
    try:
        # Try different possible paths
        logo_paths = [
            "assets/logo.png",
            "logo.png",
            "assets/logo.jpg",
            "logo.jpg"
        ]
        
        for path in logo_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        
        # If no logo found, return empty string
        return ""
    except Exception:
        return ""

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .risk-high {
        background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }

    .risk-low {
        background: linear-gradient(90deg, #51cf66, #69db7c);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }

    .sidebar .sidebar-content {
        background: #f8f9fa;
    }

    .stAlert > div {
        padding: 1rem;
        border-radius: 5px;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&display=swap');
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
</style>
""", unsafe_allow_html=True)


class EEGSeizureDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_loaded = False

    def load_model_components(self, model_path):
        """Load the trained model and preprocessors"""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"

            # Extract directory from model path
            model_dir = os.path.dirname(model_path)

            # Load model
            self.model = load_model(model_path)

            # Load preprocessors
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            encoder_path = os.path.join(model_dir, "label_encoder.joblib")

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)

            self.model_loaded = True
            return True, "Model loaded successfully!"

        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def preprocess_data(self, data):
        """Preprocess the input data for prediction"""
        try:
            # Apply scaling if scaler is available
            if self.scaler is not None:
                data_scaled = self.scaler.transform(data)
            else:
                # Basic normalization if no scaler
                data_scaled = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

            # Reshape for CNN (samples, timesteps, features)
            if len(data_scaled.shape) == 2:
                data_reshaped = data_scaled.reshape(data_scaled.shape[0], 1, data_scaled.shape[1])
            else:
                data_reshaped = data_scaled

            return data_reshaped, True, "Data preprocessed successfully"

        except Exception as e:
            return None, False, f"Error preprocessing data: {str(e)}"

    def predict(self, data):
        """Make predictions on the preprocessed data"""
        try:
            if not self.model_loaded:
                return None, None, False, "Model not loaded"

            # Preprocess data
            processed_data, success, message = self.preprocess_data(data)
            if not success:
                return None, None, False, message

            # Make predictions
            predictions_proba = self.model.predict(processed_data, verbose=0)

            # Convert probabilities to class predictions
            if predictions_proba.shape[1] == 2:  # Binary classification
                predictions = (predictions_proba[:, 1] > 0.5).astype(int)
                confidence = np.max(predictions_proba, axis=1)
            else:
                predictions = np.argmax(predictions_proba, axis=1)
                confidence = np.max(predictions_proba, axis=1)

            # Convert to labels if label encoder is available
            if self.label_encoder is not None:
                labels = self.label_encoder.inverse_transform(predictions)
            else:
                labels = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]

            return predictions, labels, confidence, True, "Predictions completed successfully"

        except Exception as e:
            return None, None, None, False, f"Error making predictions: {str(e)}"


# Initialize the detector
@st.cache_resource
def load_detector():
    return EEGSeizureDetector()


detector = load_detector()

# Header with logo and classy font
logo_base64 = load_logo()

if logo_base64:
    st.markdown(f"""
    <div class="main-header" style="background: white;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: 100px; object-fit: contain;">
            <span style="font-family: 'Cinzel', 'Times New Roman', serif; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; text-shadow: 1px 1px 3px rgba(0,0,0,0.2);">EEG Seizure Detection System</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback header without logo
    st.markdown("""
    <div class="main-header" style="background: white;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <span style="font-size: 3rem;">🧠</span>
            <span style="font-family: 'Cinzel', 'Times New Roman', serif; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; text-shadow: 1px 1px 3px rgba(0,0,0,0.2);">EEG Seizure Detection System</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 System Configuration")

    # Model loading section
    st.markdown("#### Model Loading")

    # Default model path - check multiple possible locations
    possible_model_paths = [
        "model/eeg_cnn_model.h5",
        "models/eeg_cnn_model.h5",
        "eeg_cnn_model.h5",
        "assets/eeg_cnn_model.h5"
    ]
    
    default_model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            default_model_path = path
            break
    
    if default_model_path is None:
        default_model_path = "model/eeg_cnn_model.h5"

    model_path = st.text_input(
        "Model Path:",
        value=default_model_path,
        help="Path to your trained EEG CNN model (.h5 file)"
    )

    if st.button("🔄 Load Model", type="primary"):
        with st.spinner("Loading model..."):
            success, message = detector.load_model_components(model_path)
            if success:
                st.success(message)
                st.session_state.model_loaded = True
            else:
                st.error(message)
                st.session_state.model_loaded = False

    # Model status
    if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ Model Not Loaded")

    st.markdown("---")

    # Information section
    st.markdown("#### 📋 System Information")
    st.info("""
    **Input Requirements:**
    - CSV file with preprocessed EEG data
    - Each row = 4-second EEG window
    - Features should match training data
    - No headers required
    """)

    st.markdown("#### 🎯 Prediction Classes")
    st.markdown("- **High Risk**: Seizure detected")
    st.markdown("- **Low Risk**: Normal activity")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Analytics", "📈 Visualization", "📋 Model Info"])

with tab1:
    st.markdown("### 🔍 EEG Seizure Risk Prediction")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Preprocessed EEG Data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing preprocessed EEG features"
    )

    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)

            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", data.shape[0])
            with col2:
                st.metric("Features", data.shape[1])
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            # Data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(data.head(10))

                # Basic statistics
                st.markdown("**Data Statistics:**")
                st.dataframe(data.describe())

            # Prediction section
            if st.button("🚀 Run Seizure Detection", type="primary"):
                if not detector.model_loaded:
                    st.error("❌ Please load the model first!")
                else:
                    with st.spinner("Analyzing EEG data..."):
                        # Make predictions
                        predictions, labels, confidence, success, message = detector.predict(data.values)

                        if success:
                            # Results summary
                            st.success("✅ Analysis Complete!")

                            # Calculate metrics
                            high_risk_count = np.sum([1 for label in labels if 'High' in str(label)])
                            low_risk_count = len(labels) - high_risk_count
                            avg_confidence = np.mean(confidence)

                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0; color:#dc3545;">🚨 High Risk</h4>
                                    <h2 style="margin:0; color:#dc3545;">{high_risk_count}</h2>
                                    <p style="margin:0; color:#6c757d;">Seizure segments</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0; color:#28a745;">✅ Low Risk</h4>
                                    <h2 style="margin:0; color:#28a745;">{low_risk_count}</h2>
                                    <p style="margin:0; color:#6c757d;">Normal segments</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0; color:#17a2b8;">📊 Avg Confidence</h4>
                                    <h2 style="margin:0; color:#17a2b8;">{avg_confidence:.1%}</h2>
                                    <p style="margin:0; color:#6c757d;">Model certainty</p>
                                </div>
                                """, unsafe_allow_html=True)

                            with col4:
                                risk_percentage = (high_risk_count / len(labels)) * 100
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0; color:#ffc107;">⚠️ Risk Level</h4>
                                    <h2 style="margin:0; color:#ffc107;">{risk_percentage:.1f}%</h2>
                                    <p style="margin:0; color:#6c757d;">Seizure risk</p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Detailed results
                            st.markdown("### 📋 Detailed Results")

                            # Create results dataframe
                            results_df = pd.DataFrame({
                                'Segment': range(1, len(labels) + 1),
                                'Time_Window': [f"{i * 4}-{(i + 1) * 4}s" for i in range(len(labels))],
                                'Prediction': labels,
                                'Confidence': [f"{c:.1%}" for c in confidence],
                                'Risk_Level': ['🚨 HIGH' if 'High' in str(label) else '✅ LOW' for label in labels]
                            })

                            # Display with filtering
                            filter_option = st.selectbox(
                                "Filter Results:",
                                ["All Segments", "High Risk Only", "Low Risk Only"]
                            )

                            if filter_option == "High Risk Only":
                                filtered_df = results_df[results_df['Risk_Level'] == '🚨 HIGH']
                            elif filter_option == "Low Risk Only":
                                filtered_df = results_df[results_df['Risk_Level'] == '✅ LOW']
                            else:
                                filtered_df = results_df

                            st.dataframe(filtered_df, use_container_width=True)

                            # Download results
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()

                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv_data,
                                file_name=f"eeg_seizure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                            # Store results in session state for other tabs
                            st.session_state.results_df = results_df
                            st.session_state.predictions = predictions
                            st.session_state.confidence = confidence
                            st.session_state.labels = labels

                        else:
                            st.error(f"❌ {message}")

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

with tab2:
    st.markdown("### 📊 Advanced Analytics")

    if 'results_df' in st.session_state:
        results_df = st.session_state.results_df

        # Risk distribution
        col1, col2 = st.columns(2)

        with col1:
            risk_counts = results_df['Risk_Level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                color_discrete_map={'🚨 HIGH': '#dc3545', '✅ LOW': '#28a745'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Confidence distribution
            confidence_values = [float(c.strip('%')) / 100 for c in results_df['Confidence']]
            fig_hist = px.histogram(
                x=confidence_values,
                nbins=20,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Timeline analysis
        st.markdown("#### 📈 Risk Timeline")

        # Create timeline data
        timeline_data = []
        for idx, row in results_df.iterrows():
            timeline_data.append({
                'Segment': row['Segment'],
                'Risk_Score': 1 if row['Risk_Level'] == '🚨 HIGH' else 0,
                'Confidence': float(row['Confidence'].strip('%')) / 100,
                'Time': row['Time_Window']
            })

        timeline_df = pd.DataFrame(timeline_data)

        fig_timeline = px.line(
            timeline_df,
            x='Segment',
            y='Risk_Score',
            title='Seizure Risk Over Time',
            labels={'Risk_Score': 'Risk Level (0=Low, 1=High)', 'Segment': 'Time Segment'}
        )
        fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Risk Threshold")
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Statistical summary
        st.markdown("#### 📋 Statistical Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Risk Statistics:**")
            total_segments = len(results_df)
            high_risk_segments = len(results_df[results_df['Risk_Level'] == '🚨 HIGH'])
            risk_percentage = (high_risk_segments / total_segments) * 100

            st.write(f"- Total segments analyzed: {total_segments}")
            st.write(f"- High risk segments: {high_risk_segments}")
            st.write(f"- Risk percentage: {risk_percentage:.1f}%")
            st.write(f"- Average confidence: {np.mean(st.session_state.confidence):.1%}")

        with col2:
            st.markdown("**Confidence Statistics:**")
            confidence = st.session_state.confidence
            st.write(f"- Min confidence: {np.min(confidence):.1%}")
            st.write(f"- Max confidence: {np.max(confidence):.1%}")
            st.write(f"- Std deviation: {np.std(confidence):.1%}")

            # Risk level recommendation
            if risk_percentage > 20:
                st.error("⚠️ **HIGH ALERT**: Significant seizure activity detected!")
            elif risk_percentage > 10:
                st.warning("⚠️ **MODERATE ALERT**: Some seizure activity detected")
            else:
                st.success("✅ **LOW ALERT**: Minimal seizure activity")

    else:
        st.info("📊 Upload EEG data and run analysis to view advanced analytics")

with tab3:
    st.markdown("### 📈 Data Visualization")

    if 'results_df' in st.session_state:
        results_df = st.session_state.results_df

        # Heatmap of risk over time
        st.markdown("#### 🔥 Risk Heatmap")

        # Create risk matrix (reshape data into a grid)
        segments_per_row = 10
        total_segments = len(results_df)
        rows_needed = (total_segments + segments_per_row - 1) // segments_per_row

        # Pad data to fill complete grid
        risk_values = [1 if risk == '🚨 HIGH' else 0 for risk in results_df['Risk_Level']]
        while len(risk_values) % segments_per_row != 0:
            risk_values.append(0)

        # Reshape into matrix
        risk_matrix = np.array(risk_values).reshape(-1, segments_per_row)

        fig_heatmap = px.imshow(
            risk_matrix,
            title="EEG Risk Heatmap (Red = High Risk, Blue = Low Risk)",
            color_continuous_scale=['lightblue', 'red'],
            aspect='auto'
        )
        fig_heatmap.update_layout(
            xaxis_title="Time Segments (4s windows)",
            yaxis_title="Time Blocks"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 3D visualization
        st.markdown("#### 🌐 3D Risk Landscape")

        # Create 3D surface
        x = np.arange(len(results_df))
        y = [1 if risk == '🚨 HIGH' else 0 for risk in results_df['Risk_Level']]
        z = [float(c.strip('%')) / 100 for c in results_df['Confidence']]

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=z,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Confidence")
            ),
            text=[f"Segment {i + 1}<br>Risk: {risk}<br>Confidence: {conf}"
                  for i, (risk, conf) in enumerate(zip(results_df['Risk_Level'], results_df['Confidence']))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        )])

        fig_3d.update_layout(
            title="3D Risk Analysis",
            scene=dict(
                xaxis_title="Time Segment",
                yaxis_title="Risk Level",
                zaxis_title="Confidence Score"
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Confidence vs Risk scatter
        st.markdown("#### 🎯 Confidence vs Risk Analysis")

        confidence_numeric = [float(c.strip('%')) / 100 for c in results_df['Confidence']]
        risk_numeric = [1 if risk == '🚨 HIGH' else 0 for risk in results_df['Risk_Level']]

        fig_scatter = px.scatter(
            x=confidence_numeric,
            y=risk_numeric,
            title="Confidence vs Risk Level",
            labels={'x': 'Confidence Score', 'y': 'Risk Level (0=Low, 1=High)'},
            color=risk_numeric,
            color_continuous_scale=['green', 'red']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.info("📈 Upload EEG data and run analysis to view visualizations")

with tab4:
    st.markdown("### 📋 Model Information")

    if detector.model_loaded:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🤖 Model Architecture")
            if detector.model is not None:
                # Model summary
                model_summary = []
                detector.model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))

        with col2:
            st.markdown("#### 📊 Model Details")
            if detector.model is not None:
                st.write(f"**Total Parameters:** {detector.model.count_params():,}")
                st.write(f"**Input Shape:** {detector.model.input_shape}")
                st.write(f"**Output Shape:** {detector.model.output_shape}")

                # Layer information
                st.markdown("**Layer Information:**")
                for i, layer in enumerate(detector.model.layers):
                    st.write(f"- Layer {i + 1}: {layer.__class__.__name__}")

    else:
        st.warning("⚠️ Load model to view detailed information")

    # System information
    st.markdown("#### 💻 System Information")
    st.write(f"**TensorFlow Version:** {tf.__version__}")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Usage instructions
    st.markdown("#### 📖 Usage Instructions")
    st.markdown("""
    1. **Load Model**: Enter the path to your trained model and click 'Load Model'
    2. **Upload Data**: Upload a CSV file with preprocessed EEG features
    3. **Run Analysis**: Click 'Run Seizure Detection' to analyze the data
    4. **View Results**: Check results in the Prediction, Analytics, and Visualization tabs
    5. **Download**: Export results as CSV for further analysis

    **Data Format Requirements:**
    - CSV file with numerical features only
    - Each row represents a 4-second EEG window
    - Features should match the training data format
    - No headers required (will be ignored if present)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧠 EEG Seizure Detection System v1.0 | 
        Built with Streamlit & TensorFlow | 
        For Medical Research Use
    </div>
    """,
    unsafe_allow_html=True
)
