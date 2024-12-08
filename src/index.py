# In your main.py file

from mlp import CleanData, RunML  # Assuming your classes are in your_module.py
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

if __name__ == "__main__":
    # Create an instance of CleanData
    run = CleanData()
    cleaned_data = run.data_cleaning_pipeline()

    # Create an instance of RunML (this will also run CleanData's __init__)
    deploy = RunML(cleaned_data)
    gbr_model = ('regressor', GradientBoostingRegressor(random_state=42, max_depth=5, criterion='squared_error',
                                                        learning_rate=0.1, n_estimators=50))
    task_1_result = deploy.regression_model(gbr_model)

    gbm_model = ('gbm', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.2, max_depth=5))
    task_2_result = deploy.classifier_model(gbm_model)


0