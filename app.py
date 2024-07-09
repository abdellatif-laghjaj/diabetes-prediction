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
        "SVM": SVC(),
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
    st.write(results_df)
    
    st.write("### Model Performance Comparison")
    fig = px.bar(results_df, x='Model', y='Accuracy', title='Performance des modèles', labels={'Model': 'Modèle', 'Accuracy': 'Précision'})
    st.plotly_chart(fig)

# Streamlit App
selected = option_menu(None, ["Home", "About"], icons=['house', 'info-circle'], menu_icon="cast", default_index=0, orientation='horizontal')

if selected == "Home":
    st.sidebar.title('Options de Prédiction')
    algorithm = st.sidebar.selectbox('Choisissez un modèle', ('Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'))
    show_real_time_predictions = st.sidebar.checkbox('Afficher les prédictions en temps réel')

    st.title('Prédiction du Risque de Diabète')

    st.write("## Chargement des Données")
    df = load_data()
    st.write("### Aperçu des Données")
    st.write(df.head())

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = train_models(X_train, y_train, X_test, y_test)

    st.write("## Résultats des Modèles")
    visualize_results(results)

    if show_real_time_predictions:
        st.write("## Prédictions en Temps Réel")
        model = results[algorithm]['model']
        
        st.sidebar.write("### Entrer les données")
        input_data = {}
        for column in X.columns:
            input_data[column] = st.sidebar.number_input(f'Valeur de {column}', min_value=float(X[column].min()), max_value=float(X[column].max()), value=float(X[column].mean()))
        
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        
        if st.sidebar.button('Prédire'):
            prediction = model.predict(input_df)

            # If positive make it green, else make it red
            if prediction[0] == 1:
                st.success('### Prédiction : Positif')
            else:
                st.error('### Prédiction : Négatif')

elif selected == "About":
    st.title('À propos')
    st.write("""
        ## Application de Prédiction du Risque de Diabète
        Cette application utilise des algorithmes de machine learning pour prédire le risque de diabète basé sur des données médicales.
        
        ### Auteur
        - **Nom:** Abdellatif Laghjaj
        - **Github:** [github.com/abdellatif-laghjaj](https://github.com/abdellatif-laghjaj)
    """)