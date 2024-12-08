# In your main.py file

from mlp import CleanData, RunML  # Assuming your classes are in your_module.py
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

if __name__ == "__main__":
    # Create an instance of CleanData
    run = CleanData()

    # calls data_cleaning_pipeline method to  clean and process the data
    cleaned_data = run.data_cleaning_pipeline()

    # create an instance of RunML with the cleaned data as argument
    deploy = RunML(cleaned_data)

    #for predicting temperature
    # a tuple with the name of the model as first element, second element as the ML model with its hyperparameter
    gbr_model = ('regressor', GradientBoostingRegressor(random_state=42, max_depth=5, criterion='squared_error',
                                                        learning_rate=0.1, n_estimators=50))
    #to run the model with the tuple as argument
    task_1_result = deploy.regression_model(gbr_model)

    # for predicting Plant Type-Stage
    # a tuple with the name of the model as first element, second element as the ML model with its hyperparameter
    gbm_model = ('gbm', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.2, max_depth=5))

    #to run the model with the tuple as argument
    task_2_result = deploy.classifier_model(gbm_model)
