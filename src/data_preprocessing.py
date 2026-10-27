import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath='data/nhanes_gallbladder_cohort.csv'):
    df = pd.read_csv(filepath)
    # Target column
    y = df['Gallstone_Disease_Status'].values
    X = df.drop(columns=['Gallstone_Disease_Status'])
    return X, y, df
