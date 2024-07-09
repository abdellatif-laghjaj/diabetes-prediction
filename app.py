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
    st.title('Diabetes Risk Prediction App')

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
    st.title('About')
    st.write("""
        ## Diabetes Risk Prediction Application
        This application uses machine learning algorithms to predict diabetes risk based on medical data.
        
        The app compares multiple machine learning models and automatically selects the best performing one for predictions.
        
        ### Author
        - **Name:** Abdellatif Laghjaj
        - **Github:** [github.com/abdellatif-laghjaj](https://github.com/abdellatif-laghjaj)
    """)