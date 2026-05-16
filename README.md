# Advanced AI Titanic Predictor

An internship-level, production-ready Streamlit web application that predicts passenger survival on the Titanic using an advanced, fully automated Machine Learning pipeline. 

## Advanced Features
- **Modular Architecture**: Code is split into `utils/` for preprocessing, visualization, and prediction logic.
- **Automated ML Pipeline**: Trains Logistic Regression, Random Forest, Decision Tree, KNN, and SVM models simultaneously, automatically selecting the best one based on F1 Score.
- **Interactive Plotly Visualizations**: Modern, dynamic charts for Exploratory Data Analysis and Feature Importance.
- **Real-Time Prediction Engine**: A sleek, custom CSS-styled dashboard that accepts user inputs and returns animated prediction cards with confidence probabilities.
- **Comprehensive Reporting**: Detailed model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC), Confusion Matrix heatmap, and downloadable prediction CSV/PDF reports.
- **Professional UI/UX**: Dark mode by default with an optional theme switcher, sidebar navigation, loading spinners, and metric cards.

## Project Structure
```text
codsoft_titanic_prediction/
│
├── app.py                      # Main Streamlit dashboard
├── model/                      # Stores the trained .pkl models and encoders
├── dataset/                    # Stores the Titanic CSV dataset
├── utils/
│   ├── preprocessing.py        # Data cleaning, imputation, and encoding
│   ├── visualization.py        # Plotly graph generation
│   └── prediction.py           # ML model training and evaluation
├── assets/
│   └── styles.css              # Custom dashboard styling
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## Installation & Local Execution
1. Clone the repository and navigate to the project directory:
   ```bash
   cd codsoft_titanic_prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Deployment
### Streamlit Community Cloud
1. Push this complete project folder to a public or private GitHub repository.
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app, link your GitHub account, and select the repository.
4. Set the Main file path to `app.py` and click **Deploy**.

### Render Deployment
1. Ensure your GitHub repository contains all files, including `requirements.txt`.
2. Log into [Render.com](https://render.com/).
3. Create a new **Web Service** and connect your repository.
4. Set the Build Command to `pip install -r requirements.txt` and the Start Command to `streamlit run app.py --server.port $PORT`.
5. Click **Create Web Service**.
