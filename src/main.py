# In your main.py file

from mlp import CleanData, MLPipeline  # Assuming your classes are in your_module.py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Create an instance of CleanData
run = CleanData()

# calls data_cleaning_pipeline method to  clean and process the data
cleaned_data = run.data_cleaning_pipeline()

# create an instance of RunML with the cleaned data as argument
deploy = MLPipeline(cleaned_data)

# for predicting temperature
# a tuple with the name of the model as first element, second element as the ML model with its hyperparameter
rfr_model = ('rfr', RandomForestRegressor(random_state=42, max_depth=5, min_samples_split=10, n_estimators=50))
# to run the model with the tuple as argument
task_2a_result = deploy.regression_model(rfr_model)

# for predicting Plant Type-Stage
# a tuple with the name of the model as first element, second element as the ML model with its hyperparameter
rfc_model = ('rfc', RandomForestClassifier(random_state=42, n_estimators=100))

# to run the model with the tuple as argument
task_2b_result = deploy.classification_model(rfc_model)

