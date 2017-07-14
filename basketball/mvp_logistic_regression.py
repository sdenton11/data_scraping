import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('../basketball/mvp_totals_2017.csv',
                        encoding='latin-1')

log_df = df[['MVP', 'AST%', 'BLK%', 'DRB%', 'ORB%', 'TOV%', 'eFG%']]

print (log_df.columns.values)

for column in log_df:
    log_df[column] = pd.to_numeric(log_df[column], errors='coerce')
log_df = log_df.dropna()

log_df['intercept'] = 1

train_cols = log_df.columns[1:]

print (train_cols)

logit = sm.Logit(log_df['MVP'], log_df[train_cols])

result = logit.fit()

print (result.summary())