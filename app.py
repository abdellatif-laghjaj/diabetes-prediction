import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Function to load data
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_data_upload.csv')

# Function to preprocess data
def preprocess_data(df):
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
    X = df.drop('class', axis=1)
    y = df['class']
    return X, y

# Function to train models and return results
def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    return results

# Function to visualize results
def visualize_results(results):
    results_df = pd.DataFrame([(name, data['accuracy']) for name, data in results.items()], columns=['Model', 'Accuracy'])
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    
    fig = px.bar(results_df, x='Model', y='Accuracy', 
                 title='Model Performance Comparison',
                 labels={'Model': 'Model', 'Accuracy': 'Accuracy'},
                 color='Model',
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(xaxis_title='Model', yaxis_title='Accuracy')
    st.plotly_chart(fig)

    st.write("### Detailed Results")
    st.dataframe(results_df.style.format({'Accuracy': '{:.2%}'}))

# Streamlit App
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

selected = option_menu(None, ["Home", "About"], icons=['house', 'info-circle'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected == "Home":
    st.title('Diabetes Risk Prediction App 🩺')

    # Main content
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = train_models(X_train, y_train, X_test, y_test)

    st.write("## Model Benchmarking")
    visualize_results(results)

    # Get the best performing model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']

    st.write(f"## Best Performing Model: {best_model_name}")
    st.write(f"Accuracy: {results[best_model_name]['accuracy']:.2%}")

    # Sidebar for real-time predictions
    st.sidebar.title('Real-time Prediction')
    st.sidebar.write(f"Using model: {best_model_name}")

    st.sidebar.write("### Enter patient data")
    input_data = {}
    for column in X.columns:
        input_data[column] = st.sidebar.number_input(f'{column}', min_value=float(X[column].min()), max_value=float(X[column].max()), value=float(X[column].mean()))

    if st.sidebar.button('Predict'):
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        prediction = best_model.predict(input_df)
        probability = best_model.predict_proba(input_df)[0][1]
        
        # Display prediction
        st.write("### Prediction")
        if prediction[0] == 1:
            st.error(f'Prediction: Positive (Risk: {probability:.2%})')
        else:
            st.success(f'Prediction: Negative (Risk: {probability:.2%})')

elif selected == "About":
    st.title('About the Diabetes Risk Prediction Application 🩺')
    st.write("""
        ## Overview
        This application utilizes advanced machine learning algorithms to predict the risk of diabetes based on medical data. It provides a user-friendly interface for both healthcare professionals and individuals to assess diabetes risk quickly and accurately.

        ## Key Features
        1. **Multi-Model Comparison**: The app benchmarks multiple machine learning models including Logistic Regression, Decision Tree, Random Forest, SVM, and KNN.
        2. **Automated Best Model Selection**: It automatically selects the best-performing model for predictions, ensuring optimal accuracy.
        3. **Real-time Predictions**: Users can input patient data and receive instant risk assessments.
        4. **Interactive Visualizations**: The app provides clear, interactive charts to visualize model performance.
        5. **User-friendly Interface**: With an intuitive design, the app is accessible to both medical professionals and the general public.

        ## How It Works
        1. The app loads and preprocesses the diabetes dataset.
        2. It trains and evaluates multiple machine learning models.
        3. The best-performing model is automatically selected for predictions.
        4. Users can input patient data via the sidebar for real-time risk assessment.

        ## Setup Instructions
        To run this application on your local machine:

        1. Clone the repository:
           ```
           git clone git@github.com:abdellatif-laghjaj/diabetes-prediction.git
           cd diabetes-prediction
           ```

        2. Create a virtual environment (optional but recommended):
           ```
           python -m venv venv
           source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
           ```

        3. Install the required packages:
           ```
           pip install -r requirements.txt
           ```

        4. Run the Streamlit app:
           ```
           streamlit run app.py
           ```

        5. Open your web browser and go to `http://localhost:8501` to view the app.

        ## Data Source
        The app uses the Early stage diabetes risk prediction dataset from the UCI Machine Learning Repository. Here is the [link to the dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset).

        ## Author
        - **Name:** Abdellatif Laghjaj
        - **Github:** [github.com/abdellatif-laghjaj](https://github.com/abdellatif-laghjaj)

        ## Feedback and Contributions
        Your feedback and contributions are welcome! Please feel free to submit issues or pull requests on the GitHub repository.
    """)