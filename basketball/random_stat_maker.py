import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('../basketball/mvp_totals.csv',
                        encoding='latin-1')

df = df[df['AST'] > 100]
df = df[df['FGA'] > 200]
df['AST/FGA'] = df['AST/G']/df['FGA/G']
df = df.sort("AST/FGA", ascending=False)

df = df[['name', 'Season', 'AST/FGA', 'FGA', 'AST']]

df.to_csv("FGA_to_AST.csv")