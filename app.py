# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder # FIX: Added LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix
)

# LIME - Model Interpretation
from lime.lime_tabular import LimeTabularExplainer

# --- Page Configuration and Custom Styling ---
st.set_page_config(
    layout="wide",
    page_title="Universal Data Analytics & ML App",
    page_icon="🤖"
)

# Custom CSS for a purple-themed UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        color: #f0f0f0;
    }
    .reportview-container {
        background: #1a1a2e; /* Dark purple/blue background */
    }
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0b0ff; /* Lighter purple for headings */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #2b2b4e;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        margin: 0 5px;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3b3b6e;
    }
    .stTabs [aria-selected="true"] button {
        background-color: #4b4b8e;
        color: #e0b0ff !important;
    }
    .stButton>button {
        background-color: #8a2be2; /* Blue-violet button color */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #9370db; /* Medium purple on hover */
    }
    .stDataFrame {
        font-size: 0.9em;
    }
    .sidebar .sidebar-content {
        background: #2b2b4e; /* Darker sidebar */
        padding: 2rem;
        border-radius: 10px;
    }
    .css-1f91_x{
        color: #e0b0ff;
    }
    .st-emotion-cache-1wvj576 {
        background-color: #1a1a2e;
    }
    .st-emotion-cache-1wvj576 > div {
        background-color: #2b2b4e;
        border-radius: 10px;
    }
    .css-1wiv00p { /* Target the Streamlit info box */
        background-color: #3e3e6e !important;
        border-left: 5px solid #8a2be2 !important;
    }
    .css-1wvj576, .css-1a32fsj, .css-17l4a1n, .css-1dp5x4i { /* Adjust Streamlit elements for dark theme */
        background-color: #1a1a2e;
    }
    .css-1j4f6f7 { /* Tabs wrapper */
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("Data Analytics & ML App Using Lime 📊")
st.markdown("Easily explore your data and build predictive models with one powerful tool.")
st.markdown("---")

# --- Sidebar for Uploads and Settings ---
with st.sidebar:
    st.header("⚙️ App Configuration")
    st.markdown("### 1. Upload Data")
    uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"])

    st.markdown("---")
    st.markdown("### 2. Model Settings")
    
    if uploaded_file:
        try:
            # --- FILE TYPE HANDLING ---
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload an Excel or CSV file.")
                st.stop()
            # --- END FILE TYPE HANDLING ---

            target_col = st.selectbox("Select Target Column", ["-- choose --"] + df.columns.tolist())
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
            st.stop()
    else:
        target_col = None
        st.info("💡 Please upload an Excel or CSV file to get started.")

    if target_col and target_col != "-- choose --":
        # Auto-detect logic for task type
        task_type_option = st.radio("Task Type", ["Auto-detect", "Classification", "Regression"], index=0, help="Classification is for predicting categories, Regression for continuous values.")
        
        if task_type_option == "Auto-detect":
            # Heuristic to auto-detect task type
            if df[target_col].dtype.kind in 'biufc' and df[target_col].nunique() < 20:
                task_type = "Classification"
            elif df[target_col].dtype.kind in 'biufc':
                task_type = "Regression"
            else:
                task_type = "Classification"
        else:
            task_type = task_type_option
        
        st.success(f"Task detected: **{task_type}**")

        # Model selection based on task type
        if task_type == "Classification":
            model_choice = st.selectbox("Choose a Classification Model", ["Random Forest Classifier", "Logistic Regression"])
        else:
            model_choice = st.selectbox("Choose a Regression Model", ["Random Forest Regressor", "Linear Regression"])

        # Train-test split settings
        st.markdown("---")
        st.markdown("### 3. Training Options")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5, help="Percentage of data to reserve for testing the model's performance.")
        random_state = st.number_input("Random Seed", value=42, step=1, help="A fixed number for reproducibility of results.")

        run_button = st.button("🚀 Train Model & Get Insights", type="primary")
    else:
        run_button = False

# --- Main App Logic ---
if uploaded_file is None:
    st.info("Awaiting file upload to begin...")
    st.subheader("Example Dataset Format")
    st.write("Your file should have rows as observations and columns as features. The target variable should be one of the columns.")
    st.dataframe(pd.DataFrame({
        "age": [25, 32, 40, 28],
        "salary": [55000, 72000, 85000, 60000],
        "department": ["Sales", "Engineering", "HR", "Engineering"],
        "attrition_target": [0, 1, 0, 0]
    }))
    st.stop()

# --- Initialize Session State ---
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["📊 Data Preview & EDA", "🧠 Model Training & Evaluation", "🔍 Explain Predictions (LIME)"])

# --- Tab 1: Data Preview & EDA ---
with tab1:
    st.header("Data Preview")
    st.dataframe(df.head(200))

    st.header("Exploratory Data Analysis (EDA)")
    
    eda_col1, eda_col2 = st.columns((1, 1))
    
    with eda_col1:
        with st.expander("Dataset Summary Statistics", expanded=False):
            st.write(df.describe(include='all').T)
        with st.expander("Missing Values Count", expanded=False):
            missing_vals = df.isna().sum()
            missing_df = missing_vals[missing_vals > 0].sort_values(ascending=False).to_frame('missing_count')
            if not missing_df.empty:
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values found!")

    with eda_col2:
        st.subheader("Quick Visualizations")
        plot_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Scatter Plot"])
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if plot_type == "Histogram" and numeric_cols:
            col = st.selectbox("Select column for histogram", numeric_cols)
            if col:
                fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=30, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Box Plot" and numeric_cols:
            col = st.selectbox("Select column for box plot", numeric_cols)
            if col:
                fig = px.box(df, y=col, title=f"Box Plot of {col}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Scatter Plot" and len(df.columns) >= 2:
            x_col = st.selectbox("Select X-axis column", df.columns, index=0)
            y_col = st.selectbox("Select Y-axis column", df.columns, index=1)
            color_col = st.selectbox("Select Color column (optional)", ["None"] + df.columns.tolist())
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=None if color_col == "None" else color_col,
                                 title=f"Scatter Plot: {y_col} vs. {x_col}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Not enough numeric columns for a chart.")

# --- Preprocessing Pipeline Builder ---
def build_pipeline(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor, num_cols, cat_cols

# --- Model Training Logic ---
if run_button:
    if target_col == "-- choose --":
        st.warning("Please select a target column to start modeling.")
    else:
        with st.spinner("⏳ Preparing data and training model... This might take a moment."):
            # 1. Prepare Data
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()

            # --- FIX START ---
            # Encode target variable if it's categorical for classification
            if task_type == "Classification" and y.dtype == 'object':
                st.write("Target variable is categorical. Applying Label Encoding...")
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
                st.session_state['label_encoder'] = le # Save for later use
            # --- FIX END ---
            
            y = y.loc[X.index] # Ensure alignment

            # Drop rows with missing target
            notna_mask = y.notna()
            X = X[notna_mask]
            y = y[notna_mask]
            
            selected_features = X.columns.tolist()
            X = X[selected_features]

            # 2. Build Pipeline
            preprocessor, num_cols, cat_cols = build_pipeline(X)
            
            # 3. Choose Model
            if task_type == "Classification":
                model = RandomForestClassifier(random_state=random_state) if "Random Forest" in model_choice else LogisticRegression(max_iter=1000)
            else:
                model = RandomForestRegressor(random_state=random_state) if "Random Forest" in model_choice else LinearRegression()

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # 4. Split and Train
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100.0, random_state=random_state, stratify=(y if task_type=="Classification" and y.nunique() > 1 else None)
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Store results in session state
            st.session_state['model_trained'] = True
            st.session_state['pipeline'] = pipeline
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['task_type'] = task_type
            st.session_state['selected_features'] = selected_features
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.session_state['target_col'] = target_col
        st.success("✅ Model training complete! Navigate to the other tabs to see results.")
        st.rerun()

# --- Tab 2: Model Training & Evaluation ---
with tab2:
    if not st.session_state['model_trained']:
        st.info("Click 'Train Model' in the sidebar to see results here.")
    else:
        st.header("Model Performance on Test Set")
        
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        pipeline = st.session_state['pipeline']
        X_test = st.session_state['X_test']
        task_type = st.session_state['task_type']

        res_col1, res_col2 = st.columns((1, 2))

        with res_col1:
            st.subheader("Key Metrics")
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("Precision (weighted)", f"{prec:.4f}")
                st.metric("Recall (weighted)", f"{rec:.4f}")
                st.metric("F1 Score (weighted)", f"{f1:.4f}")
                if hasattr(pipeline.named_steps['model'], "predict_proba") and y_test.nunique() == 2:
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                    st.metric("ROC AUC", f"{auc:.4f}")
            else: # Regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.metric("R-squared (R²)", f"{r2:.4f}")
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")

        with res_col2:
            st.subheader("Performance Visuals")
            if task_type == "Classification":
                cm = confusion_matrix(y_test, y_pred)
                class_labels = sorted(y_test.unique().astype(str))
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=class_labels, y=class_labels,
                                title="Confusion Matrix", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
            else: # Regression
                fig = px.scatter(x=y_test, y=y_pred, 
                                 labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                 title="Actual vs. Predicted Values", template="plotly_dark")
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                         mode='lines', name='Ideal Fit (y=x)', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        model_in_pipeline = pipeline.named_steps['model']
        if isinstance(model_in_pipeline, (RandomForestClassifier, RandomForestRegressor)):
            st.subheader("Feature Importances")
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feature_names = ohe.get_feature_names_out(st.session_state['cat_cols'])
                num_feature_names = st.session_state['num_cols']
                feature_names = num_feature_names + list(cat_feature_names)
                importances = model_in_pipeline.feature_importances_
                imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                imp_df = imp_df.sort_values('importance', ascending=False).head(20)
                fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                             title="Top 20 Feature Importances", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not generate feature importance plot. Error: {e}")

        with st.expander("Show Advanced Training Details"):
            st.write("Model:", pipeline.named_steps['model'])
            st.write("Training Set Size:", len(st.session_state['X_train']))
            st.write("Test Set Size:", len(st.session_state['X_test']))
            st.write("Numeric Features Processed:", st.session_state['num_cols'])
            st.write("Categorical Features Processed:", st.session_state['cat_cols'])
            if st.button("Download Test Set Predictions"):
                out_df = st.session_state['X_test'].copy().reset_index(drop=True)
                out_df["actual_target"] = st.session_state['y_test'].reset_index(drop=True)
                out_df["predicted_target"] = st.session_state['y_pred']
                towrite = BytesIO()
                out_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                b64 = base64.b64encode(towrite.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx">Download predictions.xlsx</a>'
                st.markdown(href, unsafe_allow_html=True)

# --- Tab 3: Explain Predictions (LIME) ---
with tab3:
    if not st.session_state.get('model_trained'):
        st.info("Train a model first to use the LIME explainer.")
    else:
        st.header("Local Interpretable Model-agnostic Explanations (LIME)")
        st.markdown("Select an instance from the test set to understand why the model made a specific prediction for it.")

        # --- Load objects from session state ---
        pipeline = st.session_state['pipeline']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        task_type = st.session_state['task_type']
        selected_features = st.session_state['selected_features']
        cat_cols = st.session_state['cat_cols']

        try:
            # --- Robust prediction function for LIME ---
            model_only = pipeline.named_steps['model']

            def robust_predict_fn(x):
                if task_type == "Classification":
                    return model_only.predict_proba(x)
                else:
                    return model_only.predict(x)


            # --- Define class names correctly for the explainer ---
            class_names = None
            if task_type == "Classification":
                if 'label_encoder' in st.session_state:
                    class_names = st.session_state['label_encoder'].classes_.astype(str)
                else:
                    class_names = [str(c) for c in np.unique(y_train)]

            # --- Initialize the LIME Explainer ---
            categorical_features_indices = [selected_features.index(col) for col in cat_cols if col in selected_features]
            
            # Get preprocessed data (the actual input to the model)
            X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
            X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

            # Get feature names after transformation
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            num_feature_names = st.session_state['num_cols']
            all_feature_names = list(num_feature_names) + list(cat_feature_names)

            explainer = LimeTabularExplainer(
                training_data=X_train_transformed,
                feature_names=all_feature_names,
                class_names=class_names,
                mode=task_type.lower(),
                discretize_continuous=False,
                random_state=42
            )

            
            # --- UI for selecting an instance ---
            instance_idx = st.selectbox(
                "Choose an instance from the test set to explain:",
                X_test.index,
                help="This is the row number from the original dataset."
            )
            
            st.write("Explaining the following instance:")
            st.dataframe(X_test.loc[[instance_idx]])

            # --- Display model prediction for context ---
            st.write("---")
            st.subheader("Model Prediction for this Instance")
            instance_pred_encoded = pipeline.predict(X_test.loc[[instance_idx]])[0]
            actual_value_encoded = st.session_state['y_test'].loc[instance_idx]

            instance_pred = instance_pred_encoded
            actual_value = actual_value_encoded

            # Decode the prediction and actual value if a label encoder was used
            if task_type == "Classification" and 'label_encoder' in st.session_state:
                le = st.session_state['label_encoder']
                instance_pred = le.inverse_transform([int(instance_pred_encoded)])[0]
                actual_value = le.inverse_transform([int(actual_value_encoded)])[0]

            pred_col, actual_col = st.columns(2)
            pred_col.metric("Predicted Value", str(instance_pred))
            actual_col.metric("Actual Value", str(actual_value))
            st.write("---")
            
            # --- Generate and display explanation on button click ---
            if st.button("Generate LIME Explanation", type="primary"):
                instance_data = X_test_transformed[X_test.index.get_loc(instance_idx)]
                
                with st.spinner("⏳ Generating LIME explanation..."):
                    explanation = explainer.explain_instance(
                        instance_data,
                        robust_predict_fn,
                        num_features=min(10, len(selected_features)),
                        # FIX: For classification, focus on the predicted class
                        top_labels=1 if task_type == "Classification" else None
                    )
                    
                    st.subheader("LIME Explanation Plot")
                    st.info("This chart shows the features that were most influential for this specific prediction. **Green** features pushed the prediction **higher** (or towards the predicted class), while **Red** features pushed it **lower**.")
                    
                    # FIX: Get the label to explain (for classification)
                    label_to_explain = explanation.top_labels[0] if task_type == "Classification" else None
                    
                    fig = explanation.as_pyplot_figure(label=label_to_explain)
                    fig.set_size_inches(4, 2)
                    st.pyplot(fig)

                    st.subheader("Explanation Details")
                    st.write("The table below lists the features and their contribution weights.")
                    
                    explanation_list = explanation.as_list(label=label_to_explain)
                    exp_df = pd.DataFrame(explanation_list, columns=['Feature Condition', 'Contribution Weight'])
                    st.dataframe(exp_df)
        
        except Exception as e:
            st.error(f"⚠️ Failed to generate LIME explanation. Error: {e}")
            st.error("This can happen if data types are inconsistent or if a selected instance has missing values that cause issues during explanation.")
