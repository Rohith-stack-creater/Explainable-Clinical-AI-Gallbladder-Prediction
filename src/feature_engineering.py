import pandas as pd
import numpy as np

def engineer_ratios(df):
    data = df.copy()
    data['Bilirubin_Ratio'] = np.round(data['Direct_Bilirubin_mg_dL'] / (data['Total_Bilirubin_mg_dL'] + 1e-6), 3)
    data['De_Ritis_Ratio'] = np.round(data['AST_IU_L'] / (data['ALT_IU_L'] + 1e-6), 2)
    data['Non_HDL_Cholesterol'] = np.round(data['Total_Cholesterol_mg_dL'] - data['HDL_mg_dL'], 1)
    return data
