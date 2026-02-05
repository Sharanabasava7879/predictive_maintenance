import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from model_training import PredictiveMaintenanceModel

# Page configuration
st.set_page_config(
    page_title="Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        color: #000000;
    }
    .metric-card h3 {
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .metric-card p {
        color: #333333;
        font-size: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
        color: #856404;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'model' not in st.session_state:
    st.session_state.model = PredictiveMaintenanceModel()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'custom_threshold' not in st.session_state:
    st.session_state.custom_threshold = 0.5

def main():
    st.markdown("<h1 class='main-header'>⚙️ Induction Motor Predictive Maintenance System</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Navigation")
        page = st.radio("Go to", 
                       ["🏠 Home", "📊 Train Model", "🔮 Predict", "📈 Analytics", "🎚️ Threshold Tuning", "📊 Baseline Comparison"],
                       index=0)
        
        st.markdown("---")
        st.info("**System Info**\n\nAlgorithm: Hybrid Ensemble\n\nEstimators: 750\n\nHandles: Imbalanced Data")
        
        if st.session_state.model_trained:
            st.success("✅ Model is Trained")
            st.metric("Current Threshold", f"{st.session_state.custom_threshold:.4f}")
        else:
            st.warning("⚠️ Model Not Trained")
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Train Model":
        show_training()
    elif page == "🔮 Predict":
        show_prediction()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "🎚️ Threshold Tuning":
        show_threshold_tuning()
    elif page == "📊 Baseline Comparison":
        show_baseline_comparison()

def show_home():
    """Home page with overview"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Failure Prediction</h3>
            <p>Predict motor failures before they occur using advanced ML algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Real-time Analytics</h3>
            <p>Monitor motor health with comprehensive analytics dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 System Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Machine Learning Features
        - ✅ Hybrid Ensemble Algorithm
        - ✅ SMOTE-Tomek for imbalanced data
        - ✅ Advanced feature engineering
        - ✅ Real-time failure probability
        - ✅ Manual threshold optimization
        """)
    
    with col2:
        st.markdown("""
        #### User Interface Features
        - ✅ Interactive dashboard
        - ✅ Real-time predictions
        - ✅ Visual performance metrics
        - ✅ Custom threshold tuning
        - ✅ Comprehensive cost analysis
        """)
    
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    st.markdown("""
    1. **📊 Train Model** - Upload your dataset and train
    2. **🎚️ Threshold Tuning** - Fine-tune decision threshold
    3. **🔮 Predict** - Enter motor parameters
    4. **📈 Analytics** - View model performance
    """)

def show_training():
    """Model training page"""
    st.header("📊 Model Training")
    
    st.info("💡 Upload your motor dataset to train the model")
    
    uploaded_file = st.file_uploader("📁 Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success(f"✅ Dataset loaded! Shape: {df.shape}")
            
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            with st.expander("📊 Statistics", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                failures = df['Machine failure'].sum()
                non_failures = len(df) - failures
                
                with col1:
                    st.metric("Total", len(df))
                with col2:
                    st.metric("Failures", failures)
                with col3:
                    st.metric("Non-Failures", non_failures)
                with col4:
                    st.metric("Ratio", f"{non_failures/max(failures,1):.1f}:1")
                
                fig = px.pie(
                    values=[non_failures, failures],
                    names=['No Failure', 'Failure'],
                    title='Class Distribution',
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
            with col2:
                handle_imbalance = st.checkbox("Handle Imbalance (SMOTE-Tomek)", value=True)
            
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                train_model(df, test_size, handle_imbalance)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def train_model(df, test_size, handle_imbalance):
    """Train the model"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(10)
        df_processed = st.session_state.preprocessor.create_features(df)
        
        status_text.text("📋 Preparing data...")
        progress_bar.progress(20)
        X, y, feature_columns = st.session_state.preprocessor.prepare_data(df_processed)
        st.session_state.feature_columns = feature_columns
        
        status_text.text("✂️ Splitting data...")
        progress_bar.progress(30)
        X_train, X_test, y_train, y_test = st.session_state.preprocessor.split_and_scale(
            X, y, test_size=test_size
        )
        
        status_text.text("🤖 Training failure prediction model...")
        progress_bar.progress(40)
        st.session_state.model.train_failure_model(X_train, y_train, handle_imbalance)
        progress_bar.progress(70)
        
        status_text.text("📊 Evaluating...")
        results = st.session_state.model.evaluate_model(X_test, y_test)
        progress_bar.progress(90)
        
        # Store predictions for threshold tuning
        y_pred_proba = st.session_state.model.failure_model.predict_proba(X_test)[:, 1]
        st.session_state.y_test = y_test
        st.session_state.y_pred_proba = y_pred_proba
        
        status_text.text("💾 Saving models...")
        os.makedirs('models', exist_ok=True)
        st.session_state.model.save_models()
        st.session_state.preprocessor.save_scaler()
        progress_bar.progress(100)
        
        status_text.text("✅ Complete!")
        
        st.session_state.model_trained = True
        st.session_state.test_results = results
        st.session_state.custom_threshold = results['optimal_threshold']
        
        st.balloons()
        st.success("🎉 Model trained successfully!")
        
        st.markdown("---")
        st.subheader("📈 Initial Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        with col2:
            st.metric("F1 Score", f"{results['f1']:.4f}")
        with col3:
            st.metric("Precision", f"{results['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{results['recall']:.4f}")
        
        st.info("💡 Go to **Threshold Tuning** to optimize the decision threshold!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_threshold_tuning():
    """Interactive threshold tuning page"""
    st.header("🎚️ Threshold Tuning")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first!")
        return
    
    st.markdown("""
    <div class='info-box'>
        <h4>💡 About Threshold Tuning</h4>
        <p>The threshold determines when the model predicts a failure. Adjust it to balance:</p>
        <ul>
            <li><b>Lower threshold</b> → Catch more failures (higher recall) but more false alarms</li>
            <li><b>Higher threshold</b> → Fewer false alarms (higher precision) but may miss failures</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Threshold slider
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        threshold = st.slider(
            "🎯 Decision Threshold",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state.custom_threshold,
            step=0.01,
            help="Adjust threshold to see impact on performance metrics"
        )
    
    with col2:
        if st.button("🔄 Reset to Optimal", use_container_width=True):
            threshold = st.session_state.test_results['optimal_threshold']
            st.session_state.custom_threshold = threshold
            st.rerun()
    
    with col3:
        if st.button("💾 Save Threshold", type="primary", use_container_width=True):
            st.session_state.custom_threshold = threshold
            st.session_state.model.optimal_threshold = threshold
            st.success(f"✅ Saved: {threshold:.4f}")
    
    # Calculate metrics for current threshold
    y_test = st.session_state.y_test
    y_pred_proba = st.session_state.y_pred_proba
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) if (tp + fp) > 0 else 0
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    total_failures = sum(y_test)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # Display metrics
    st.markdown("---")
    st.subheader("📊 Performance at Current Threshold")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_f1 = f1 - st.session_state.test_results['f1']
        st.metric("F1 Score", f"{f1:.4f}", delta=f"{delta_f1:.4f}")
    
    with col2:
        delta_prec = precision - st.session_state.test_results['precision']
        st.metric("Precision", f"{precision:.4f}", delta=f"{delta_prec:.4f}")
    
    with col3:
        delta_rec = recall - st.session_state.test_results['recall']
        st.metric("Recall", f"{recall:.4f}", delta=f"{delta_rec:.4f}")
    
    with col4:
        delta_acc = accuracy - st.session_state.test_results['accuracy']
        st.metric("Accuracy", f"{accuracy:.4f}", delta=f"{delta_acc:.4f}")
    
    with col5:
        st.metric("Threshold", f"{threshold:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix and Key Metrics side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Failure', 'Failure'],
                       y=['No Failure', 'Failure'],
                       text_auto=True,
                       color_continuous_scale='Blues',
                       aspect='auto')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Metrics")
        
        # Status indicators
        fn_status = "🎉 PERFECT!" if fn == 0 else "✅ EXCELLENT" if fn <= 2 else "⚠️ NEEDS WORK"
        fp_status = "🎉 EXCELLENT" if fp < 100 else "✅ GOOD" if fp < 150 else "⚠️ HIGH"
        
        st.markdown(f"""
        **Confusion Matrix Breakdown:**
        - True Negatives: **{tn}** ✅
        - False Positives: **{fp}** {fp_status}
        - False Negatives: **{fn}** {fn_status}
        - True Positives: **{tp}** ✅
        
        **Detection Analysis:**
        - Total Failures: **{total_failures}**
        - Detected: **{tp}** ({tp/total_failures*100:.1f}%)
        - Missed: **{fn}** ({fn/total_failures*100:.1f}%)
        - False Positive Rate: **{fpr*100:.2f}%**
        """)
        
        # Cost Analysis
        fa_cost = fp * 100
        mf_cost = fn * 10000
        total_cost = fa_cost + mf_cost
        
        st.markdown("---")
        st.markdown(f"""
        **💰 Cost Analysis:**
        - False Alarm Cost: **${fa_cost:,}**
        - Missed Failure Cost: **${mf_cost:,}**
        - **Total Cost: ${total_cost:,}**
        """)
    
    st.markdown("---")
    
    # Interactive plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Threshold Impact on Metrics")
        
        # Calculate metrics across threshold range
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = []
        precisions = []
        recalls = []
        
        for t in thresholds:
            y_p = (y_pred_proba >= t).astype(int)
            f1_scores.append(f1_score(y_test, y_p))
            precisions.append(precision_score(y_test, y_p) if sum(y_p) > 0 else 0)
            recalls.append(recall_score(y_test, y_p))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines', name='F1 Score', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', name='Recall', line=dict(color='red')))
        
        # Add current threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="orange", annotation_text="Current")
        
        fig.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 False Positives vs False Negatives")
        
        fps = []
        fns = []
        
        for t in thresholds:
            y_p = (y_pred_proba >= t).astype(int)
            cm_t = confusion_matrix(y_test, y_p)
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            fps.append(fp_t)
            fns.append(fn_t)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=fps, mode='lines', name='False Positives', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=thresholds, y=fns, mode='lines', name='False Negatives', line=dict(color='red')))
        
        # Add current threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="blue", annotation_text="Current")
        
        fig.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Count",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation based on metrics
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if fn == 0 and fp < 150:
        st.success("🎉 **EXCELLENT BALANCE!** Zero missed failures with acceptable false alarms.")
    elif fn == 0:
        st.info(f"✅ Zero missed failures, but {fp} false alarms. Consider increasing threshold slightly to reduce false positives.")
    elif fn <= 2 and fp < 150:
        st.success(f"✅ **GOOD BALANCE!** Only {fn} missed failure(s) with {fp} false alarms.")
    elif fn > 2:
        st.warning(f"⚠️ **{fn} failures missed.** Consider lowering threshold to improve recall.")
    elif fp > 200:
        st.warning(f"⚠️ **{fp} false alarms.** Consider raising threshold to reduce false positives.")

def show_prediction():
    """Prediction page"""
    st.header("🔮 Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first!")
        return
    
    st.markdown("### Enter Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        air_temp = st.number_input("Air Temp (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.number_input("Process Temp (K)", 305.0, 315.0, 310.0, 0.1)
        speed = st.number_input("Speed (rpm)", 1000, 3000, 1500, 10)
    
    with col2:
        torque = st.number_input("Torque (Nm)", 10.0, 80.0, 40.0, 0.1)
        tool_wear = st.number_input("Tool Wear (min)", 0, 250, 100, 1)
    
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        make_prediction(air_temp, process_temp, speed, torque, tool_wear)

def make_prediction(air_temp, process_temp, speed, torque, tool_wear):
    """Make prediction"""
    try:
        temp_diff = process_temp - air_temp
        power = torque * speed
        temp_ratio = process_temp / air_temp
        torque_speed_ratio = torque / (speed + 1)
        
        tool_wear_low = 1 if tool_wear < 50 else 0
        tool_wear_medium = 1 if 50 <= tool_wear < 150 else 0
        tool_wear_high = 1 if tool_wear >= 150 else 0
        
        input_data = np.array([[
            air_temp, process_temp, speed, torque, tool_wear,
            temp_diff, power, temp_ratio, torque_speed_ratio,
            tool_wear_low, tool_wear_medium, tool_wear_high
        ]])
        
        input_scaled = st.session_state.preprocessor.scaler.transform(input_data)
        
        # Use custom threshold
        prediction, probability = st.session_state.model.predict_failure(
            input_scaled, 
            threshold=st.session_state.custom_threshold
        )
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.markdown("""
                <div class='danger-box'>
                    <h2 style='color: #dc3545; text-align: center;'>⚠️ FAILURE</h2>
                    <p style='text-align: center;'>Maintenance needed!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-box'>
                    <h2 style='color: #28a745; text-align: center;'>✅ NORMAL</h2>
                    <p style='text-align: center;'>Operating OK</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Failure Probability", f"{probability[0]*100:.2f}%")
            st.metric("Current Threshold", f"{st.session_state.custom_threshold:.4f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[0]*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red" if probability[0] > 0.7 else "orange" if probability[0] > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 70], 'color': 'lightyellow'},
                        {'range': [70, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': st.session_state.custom_threshold * 100
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction[0] == 1:
            st.error("🔴 STOP operation - immediate maintenance required")
        if probability[0] > 0.7:
            st.warning("⚠️ HIGH RISK - schedule maintenance within 24 hours")
        elif probability[0] > 0.4:
            st.warning("⚡ MODERATE RISK - monitor closely and plan maintenance")
        else:
            st.success("✅ LOW RISK - normal operation")
            
        if temp_diff > 10:
            st.info("🌡️ Temperature difference high - check cooling system")
        if tool_wear > 200:
            st.info("🔧 Tool wear exceeds 200 min - consider replacement soon")
        if torque > 60:
            st.info("⚙️ High torque detected - ensure motor is not overloaded")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_analytics():
    """Analytics page"""
    st.header("📈 Analytics")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train model first!")
        return
    
    # Use current threshold for analytics
    threshold = st.session_state.custom_threshold
    y_test = st.session_state.y_test
    y_pred_proba = st.session_state.y_pred_proba
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) if (tp + fp) > 0 else 0
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    st.subheader("🎯 Performance (Current Threshold)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
    with col2:
        st.metric("F1", f"{f1:.4f}")
    with col3:
        st.metric("Precision", f"{precision:.4f}")
    with col4:
        st.metric("Recall", f"{recall:.4f}")
    with col5:
        st.metric("Accuracy", f"{accuracy:.4f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual"),
                       x=['No Failure', 'Failure'],
                       y=['No Failure', 'Failure'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Insights")
        
        st.markdown(f"""
        **True Positives:** {tp}  
        **True Negatives:** {tn}  
        **False Positives:** {fp}  
        **False Negatives:** {fn}  
        
        **Accuracy:** {accuracy:.4f}
        **Threshold:** {threshold:.4f}
        """)
        
        if fn == 0:
            st.success("🏆 PERFECT! Zero failures missed!")
        elif fn <= 2:
            st.success(f"✅ EXCELLENT! Only {fn} failure(s) missed")
        else:
            st.warning(f"⚠️ {fn} failures missed")
    
    st.markdown("---")
    
    # Cost Analysis
    st.subheader("💰 Cost-Benefit Analysis")
    
    fa_cost = fp * 100
    mf_cost = fn * 10000
    total_cost = fa_cost + mf_cost
    max_loss = sum(y_test) * 10000
    savings = max_loss - total_cost
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("False Alarm Cost", f"${fa_cost:,}")
    with col2:
        st.metric("Missed Failure Cost", f"${mf_cost:,}")
    with col3:
        st.metric("Total Cost", f"${total_cost:,}")
    with col4:
        st.metric("Net Savings", f"${savings:,}")
    
    st.markdown("---")
    
    # ROC Curve
    st.subheader("📉 ROC Curve")
    
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_curve, y=tpr_curve, mode='lines', 
                            name=f'ROC (AUC={roc_auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                            name='Random', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate')
    st.plotly_chart(fig, use_container_width=True)

def show_baseline_comparison():
    """Baseline comparison and statistical validation page"""
    st.header("📊 Baseline Comparison & Statistical Validation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first!")
        return
    
    st.markdown("""
    <div class='info-box'>
        <h4>💡 About Baseline Comparison</h4>
        <p>Compare your Hybrid Ensemble against standard machine learning baselines:</p>
        <ul>
            <li><b>Logistic Regression</b> - Simple linear baseline</li>
            <li><b>Random Forest</b> - Tree-based ensemble</li>
            <li><b>XGBoost/Gradient Boosting</b> - Advanced gradient boosting</li>
        </ul>
        <p>Includes 10-fold cross-validation and statistical significance testing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if 'baseline_results' not in st.session_state:
        if st.button("🚀 Run Baseline Comparison", type="primary", use_container_width=True):
            with st.spinner("Running baseline comparison... This may take 5-10 minutes..."):
                try:
                    # Import the baseline comparison module
                    sys.path.append('src')
                    from baseline_comparison import run_complete_comparison
                    
                    # Get training data
                    X_train = st.session_state.preprocessor.scaler.transform(
                        st.session_state.preprocessor.prepare_data(st.session_state.df)[0]
                    )
                    
                    # Run comparison
                    comparison, results, cv_results = run_complete_comparison(
                        st.session_state.X_train if 'X_train' in st.session_state else X_train,
                        st.session_state.y_train if 'y_train' in st.session_state else st.session_state.y_test,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        your_trained_model=st.session_state.model.failure_model,
                        model_name="Hybrid Ensemble"
                    )
                    
                    st.session_state.baseline_comparison = comparison
                    st.session_state.baseline_results = results
                    st.session_state.baseline_cv_results = cv_results
                    
                    st.success("✅ Baseline comparison complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error running comparison: {str(e)}")
                    st.exception(e)
    else:
        st.success("✅ Baseline comparison already completed!")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Test Set Performance Comparison")
        
        # Create comparison DataFrame
        results_data = []
        for model_name, metrics in st.session_state.baseline_results.items():
            results_data.append({
                'Model': model_name,
                'F1 Score': f"{metrics['f1']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'FN': metrics['fn'],
                'FP': metrics['fp'],
                'Time (s)': f"{metrics['training_time']:.2f}"
            })
        
        df_results = pd.DataFrame(results_data)
        
        # Highlight best scores
        def highlight_best(s):
            if s.name in ['F1 Score', 'Recall', 'Precision', 'Accuracy', 'ROC-AUC']:
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]
            elif s.name in ['FN', 'FP']:
                is_min = s == s.min()
                return ['background-color: lightgreen' if v else '' for v in is_min]
            return [''] * len(s)
        
        st.dataframe(df_results.style.apply(highlight_best), use_container_width=True, hide_index=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics comparison
            metrics_to_plot = ['F1 Score', 'Recall', 'Precision', 'ROC-AUC']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                values = [float(row[metric]) for _, row in df_results.iterrows()]
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_results['Model'],
                    y=values
                ))
            
            fig.update_layout(
                title="Performance Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # FP and FN comparison
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                name='False Negatives',
                x=df_results['Model'],
                y=df_results['FN'],
                marker_color='red'
            ))
            
            fig2.add_trace(go.Bar(
                name='False Positives',
                x=df_results['Model'],
                y=df_results['FP'],
                marker_color='orange'
            ))
            
            fig2.update_layout(
                title="Errors Comparison (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="Count",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cross-validation results
        if 'baseline_cv_results' in st.session_state:
            st.markdown("---")
            st.subheader("🔄 10-Fold Cross-Validation Results")
            
            cv_data = []
            for model_name, metrics in st.session_state.baseline_cv_results.items():
                cv_data.append({
                    'Model': model_name,
                    'F1 Score': f"{metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}",
                    'Recall': f"{metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}",
                    'Precision': f"{metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}",
                    'ROC-AUC': f"{metrics['roc_auc']['mean']:.4f} ± {metrics['roc_auc']['std']:.4f}"
                })
            
            df_cv = pd.DataFrame(cv_data)
            st.dataframe(df_cv, use_container_width=True, hide_index=True)
            
            st.info("💡 **Mean ± Std** shows average performance across 10 folds with variability")
        
        # Statistical significance
        st.markdown("---")
        st.subheader("📈 Statistical Significance")
        
        st.markdown("""
        **Paired t-test results** comparing Hybrid Ensemble against each baseline:
        - **p < 0.05**: Statistically significant difference
        - **Cohen's d**: Effect size (small: 0.2, medium: 0.5, large: 0.8)
        """)
        
        if st.button("🔬 Run Statistical Tests", use_container_width=True):
            with st.spinner("Running statistical tests..."):
                try:
                    stat_results = st.session_state.baseline_comparison.statistical_tests(
                        metric='f1',
                        your_model_name='Hybrid Ensemble'
                    )
                    
                    if stat_results:
                        stat_data = []
                        for baseline, results in stat_results.items():
                            stat_data.append({
                                'Baseline': baseline,
                                'Mean Difference': f"{results['mean_difference']:+.4f}",
                                'p-value': f"{results['p_value']:.4f}",
                                "Cohen's d": f"{results['cohens_d']:.4f}",
                                'Significant?': '✅ Yes' if results['is_significant'] else '❌ No',
                                'Better?': '✅ Yes' if results['is_better'] else '❌ No'
                            })
                        
                        df_stat = pd.DataFrame(stat_data)
                        st.dataframe(df_stat, use_container_width=True, hide_index=True)
                        
                        # Interpretation
                        significant_better = sum(1 for r in stat_results.values() if r['is_significant'] and r['is_better'])
                        total = len(stat_results)
                        
                        if significant_better == total:
                            st.success(f"🎉 **Hybrid Ensemble is significantly better than ALL {total} baselines!**")
                        elif significant_better > 0:
                            st.info(f"✅ Hybrid Ensemble is significantly better than {significant_better}/{total} baselines")
                        else:
                            st.warning("⚠️ No significant improvement over baselines detected")
                    
                except Exception as e:
                    st.error(f"Error running statistical tests: {str(e)}")
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Test Results CSV", use_container_width=True):
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="baseline_comparison_test.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'baseline_cv_results' in st.session_state:
                if st.button("📥 Download CV Results CSV", use_container_width=True):
                    csv = df_cv.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name="baseline_comparison_cv.csv",
                        mime="text/csv"
                    )
        
        # Reset option
        st.markdown("---")
        if st.button("🔄 Run Comparison Again", use_container_width=True):
            del st.session_state.baseline_results
            del st.session_state.baseline_cv_results
            if 'baseline_comparison' in st.session_state:
                del st.session_state.baseline_comparison
            st.rerun()