## Overview
This application utilizes advanced machine learning algorithms to predict the risk of diabetes based on medical data. It provides a user-friendly interface for both healthcare professionals and individuals to assess diabetes risk quickly and accurately.

## Author
- **Name:** Abdellatif Laghjaj
- **Github:** [github.com/abdellatif-laghjaj](https://github.com/abdellatif-laghjaj)

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