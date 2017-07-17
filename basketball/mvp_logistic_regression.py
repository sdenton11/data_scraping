import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib.pyplot as plt
import patsy as pa
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('../basketball/mvp_totals.csv',
                        encoding='latin-1')

# logistic regression using statsmodels
"""
log_df = df[['MVP', 'AST%', 'Win %', 'TRB/G', 'PTS/G', 'WS']]

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
print (np.exp(result.params))
"""

# logistic regression using sklearn
# First formalize the dataframe in a way that will work
log_2df = df[['MVP', 'AST%', 'Win %', 'DWS', 'TRB/G', 'PTS/G', 'WS', 'name']]

log_2df['AST_pct'] = log_2df['AST%']
log_2df['Win_pct'] = log_2df['Win %']
log_2df['reb_g'] = log_2df['TRB/G']
log_2df['pts_g'] = log_2df['PTS/G']

log_2df = log_2df[['MVP', 'AST_pct', 'Win_pct', 'DWS', 'reb_g', 'pts_g', 'WS', 'name']]

for column in log_2df:
    if not column == 'name':
        log_2df[column] = pd.to_numeric(log_2df[column], errors='coerce')
log_2df = log_2df.dropna()

# filter out WS that were negative as the model needs positive valeus
log_2df = log_2df[log_2df['WS'] > 0]
log_2df = log_2df[log_2df['DWS'] > 0]

print (log_2df)

y, X = pa.dmatrices('MVP ~ AST_pct + Win_pct + DWS + reb_g + pts_g + WS'
                    , log_2df, return_type="dataframe")

y = np.ravel(y)

model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
(model.score(X, y))

# what percentage had success?
(y.mean())

# examine the coefficients
result = pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

result.columns = ['Variable', 'Coefficient']

scores, pvalues = chi2(X, y)

result ['p values'] = np.round(pvalues, 3)

result['Coefficient'] = result['Coefficient'].map(lambda x: x[0])

result.round({'Coefficient' : 3})

print (result)


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# predict class labels for the test set
predicted = model2.predict(X_test)
print (predicted)

# generate class probabilities
probs = model2.predict_proba(X_test)
print (probs)

# generate evaluation metrics
print (metrics.accuracy_score(y_test, predicted))
print (model.score(X, y))

player = []
for row in log_2df.iterrows():
    index, data = row
    if data[7] == 'Kawhi-Leonard':
        player.append(data.tolist())

print (player)

arr = [1]
for number in player[4][1:7]:
    arr.append(number)

# Kawhi Leonard 2017 season -- 4.9% chance of winning MVP
print (model.predict_proba(np.array(arr)))
