import pandas as pd
import numpy as np
import pickle

def recover_data(hashed_df, mapping_file='feature_mapping.pkl'):
    with open(mapping_file, 'rb') as f:
        map_data = pickle.load(f)
    
    df_recovered = hashed_df.copy()
    for col, m in map_data.items():
        if col in df_recovered.columns:
            df_recovered[col] = df_recovered[col].map(m)
    return df_recovered

if __name__ == "__main__":
    data_secret = pd.read_csv('trace_encrypted.txt',sep='\t')
    df_restored = recover_data(data_secret)
    df_restored.to_csv('trace.txt', sep='\t', index=False)