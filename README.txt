Insurance premium calculation is influenced by multiple factors such as age, Previous claims, health score, Credit score.
This project builds a regression-based ML system to predict premium amounts accurately while following MLOps best practices.

** Key highlights:
        Multiple models trained & compared
        Experiment tracking using MLflow
        Model versioning & registry via DAGsHub
        Production-ready deployment with Streamlit

** Tech Stack
        Python
        Scikit-learn
        XGBoost
        MLflow (experiment tracking & model registry)
        DAGsHub (remote MLflow tracking)
        Streamlit (web app deployment)
        Pandas / NumPy
        Matplotlib / Seaborn

** Machine Learning Workflow

Data Preprocessing

* Handling missing values
        Ordinal encoding (Education, Exercise Frequency)
        One-hot encoding (Gender, Location, Policy Type)
        Feature scaling

* Model Training
        Linear Regression
        Random Forest
        Decision Tree
        XGBoost (best performing model)

* Experiment Tracking
        Metrics logged: MAE, RMSE, R²
        Parameters & artifacts logged with MLflow
        Best model registered in MLflow Model Registry

* Model Promotion
        Best XGBoost model promoted to Production


** Streamlit Application
- The trained Production model is loaded directly from the MLflow Model Registry.

    Features:
        User-friendly input form
        Real-time premium prediction
        Uses a production-grade ML model







