import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor



class CleanData:

    def __init__(self):

        self.df_agri = None
        self.nutrient_cols = None

    def load_data(self):
        # connect to the SQLite database
        conn = sqlite3.connect('../data/agri.db')
        cursor = conn.cursor()
        # execute a query to get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # fetch all results
        tables = cursor.fetchall()
        # execute an SQL query to fetch data from the table
        query = "SELECT * FROM farm_data"
        # load query into dataframe
        self.df_agri = pd.read_sql_query(query, conn)

        return self.df_agri
    # drop Humidity Sensor
    def drop_humidity_sensor(self):
        return self.df_agri.drop(columns=['Humidity Sensor (%)'], inplace=True)
    # remove duplicates
    def drop_duplicates(self):
        return self.df_agri.drop_duplicates(inplace=True)

    def remove_units_nutrient_sensors(self):
        # get Nutrients N, P, K attributes and store them in a list
        self.nutrient_cols = list(filter(lambda col: 'Nutrient' in col, self.df_agri.columns))

        # using a for loop to loop through each column and apply a lambda function.
        for col in self.nutrient_cols:
            # for each value, if the value is None, return null value,
            # else do a split and return only the first elements of the list.
            self.df_agri[col] = self.df_agri[col].apply(lambda x: np.nan if x is None else x.split(' ')[0])
            # convert the data type to float, this is needed to plot histogram
            self.df_agri[col] = self.df_agri[col].astype('float64')

        return self.df_agri

    def remove_negative_values(self):

        def is_negative(col):
            return self.df_agri[col].min() < 0

        # using filter function to identify the target attributes and store them in a list
        columns_with_neg = list(filter(is_negative, self.df_agri.describe().columns))

        # get rows with negative values in one of the temperature, light intensity and EC sensors
        rows_with_neg_val = self.df_agri[(self.df_agri[columns_with_neg] < 0).any(axis=1)].index
        self.df_agri.drop(rows_with_neg_val, inplace=True)

        return self.df_agri

    def standardize_capitalization(self):
        # change to title case using lambda
        self.df_agri['Plant Stage'] = self.df_agri['Plant Stage'].apply(lambda x: x.title())
        self.df_agri['Plant Type'] = self.df_agri['Plant Type'].apply(lambda x: x.title())

        return self.df_agri

    def impute_null_values(self):
        # using for loop to impute mean to missing values of each nutrient attribute
        for col in self.nutrient_cols:
            # Calculate the mean of the column, ignoring NaN values
            column_mean = self.df_agri[col].mean()
            # Impute missing values with mean
            self.df_agri[col].fillna(column_mean, inplace=True)

        for col in ['Temperature Sensor (°C)', 'Light Intensity Sensor (lux)', 'Water Level Sensor (mm)']:
            # Calculate the median of the column, ignoring NaN values
            column_median = self.df_agri[col].median()
            # Impute missing values with median
            self.df_agri[col].fillna(column_median, inplace=True)

        return self.df_agri

    def remove_outlier(self):
        def check_and_remove_outliers(column):
            # get 1st & 3rd quantile of the attributes
            q1 = self.df_agri[column].quantile(0.25)
            q3 = self.df_agri[column].quantile(0.75)

            # inter-quantile range
            iqr = q3 - q1

            # define the lower and upper limits of acceptable range
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # get dataset within non outlier range
            df_no_outliers = self.df_agri[(self.df_agri[column] >= lower_bound) & (self.df_agri[column] <= upper_bound)]

            return df_no_outliers
        # get only attributes with numerical datatype
        numeric_cols = self.df_agri.select_dtypes(include=[np.number]).columns
        # write a function to remove outlier
        for column in numeric_cols:
            self.df_agri = check_and_remove_outliers(column)

        return self.df_agri

    def data_transformation(self):
        # combine the 3 attributes Nutrient N,P,K into one with their average
        self.df_agri['Nutrient NPK Sensors (ppm)'] = self.df_agri[self.nutrient_cols].mean(axis=1)

        # concat attributes Plant Type and Plant Stage as one
        self.df_agri['Plant Type-Stage'] = self.df_agri['Plant Type'] + ' ' + self.df_agri['Plant Stage']

        # remove columns
        cols_to_be_dropped = ['System Location Code', 'Previous Cycle Plant Type',
                              'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)',
                              'Nutrient K Sensor (ppm)', 'pH Sensor']

        self.df_agri.drop(columns=cols_to_be_dropped, inplace=True)

        return self.df_agri
    # orchestrates the data cleaning process by calling the individual cleaning methods sequentially
    def data_cleaning_pipeline(self):

        self.load_data()
        print('Data loaded. Cleaning Data...')
        self.drop_humidity_sensor()
        self.drop_duplicates()
        self.remove_units_nutrient_sensors()
        self.remove_negative_values()
        self.standardize_capitalization()
        self.impute_null_values()
        self.remove_outlier()
        print('Data cleaned. Transforming Data...')
        self.data_transformation()
        print('Data transformed.')

        return self.df_agri


class RunML:
    def __init__(self, df_agri):  # add df_agri parameter
        self.df_agri_cleaned = df_agri
    # regression model for predicting Temperature
    def regression_model(self, model):
        print('Running regression model...')
        # 1. Separate features (X) and target variable (y)
        X = self.df_agri_cleaned.drop(['Temperature Sensor (°C)', 'Plant Type-Stage'],
                                      axis=1)  
        y = self.df_agri_cleaned['Temperature Sensor (°C)']

        # 2. Define preprocessing steps
        # 2.1 One-hot encode the categorical feature
        categorical_feature = ['Plant Type', 'Plant Stage']  # identify the categorial features
        categorical_transformer = Pipeline(steps=[           # one-hot encode the features and drop 1 column to prevent multicollinearity
            ('onehot', OneHotEncoder(drop='first'))
        ])
        # 2.2 Scale the numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # 2.3 Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_feature)
            ])

        # 3. Create the pipeline with preprocessing and the provided model
        r_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            model
        ])

        # 4. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Train the pipeline
        r_pipeline.fit(X_train, y_train)

        # 6. Make predictions on the test set
        y_pred = r_pipeline.predict(X_test)

        # 7. Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print('Task 1a - Predict Temperature Model Result:')
        print(f"Mean Squared Error: {mse:.3f}")
        print(f"R-squared: {r2:.3f}\n")

        return r_pipeline

    def classifier_model(self, model):
        print('Running classifier model...')
        # 1. Separate features (X) and target variable (y)
        X = self.df_agri_cleaned.drop(['Plant Type-Stage', 'Plant Type',
                                       'Plant Stage'], axis=1)  # Replace 'target_variable' with the actual column name
        y = self.df_agri_cleaned['Plant Type-Stage']

        # 2. Create the pipeline with preprocessing and the provided model
        c_pipeline = Pipeline([('scaler', StandardScaler()),
                               model
                               ])

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Train the pipeline
        c_pipeline.fit(X_train, y_train)

        # 5. Make predictions on the test set
        y_pred = c_pipeline.predict(X_test)

        # 6. Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        # accuracy score and classfication report
        print('Task 1b - Predict Plant Type-Stage Model Result:')
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

        return c_pipeline
