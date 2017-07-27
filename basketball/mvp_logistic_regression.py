import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib.pyplot as plt
import patsy as pa
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import warnings

# logistic regression using statsmodels
"""
df = pd.read_csv('mvp_totals.csv', encoding='latin-1')

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

def odds_of_mvp_with_normal_stats(name, season):
    # import the dataframe of players
    df = pd.read_csv('../basketball/mvp_totals.csv',
                     encoding='latin-1')

    # logistic regression using sklearn
    # First formalize the dataframe in a way that will work
    log_2df = df[['MVP', 'AST%', 'Win %', 'DWS', 'OWS',
                  'TRB/G', 'PTS/G', 'WS', 'name', 'Season']]

    log_2df['ast_pct'] = log_2df['AST%']
    log_2df['Win_pct'] = log_2df['Win %']
    log_2df['reb_g'] = log_2df['TRB/G']
    log_2df['pts_g'] = log_2df['PTS/G']

    log_2df = log_2df[['MVP', 'ast_pct', 'Win_pct', 'DWS', 'OWS',
                       'reb_g', 'pts_g', 'WS', 'name', 'Season']]

    # convert the columns to numeric
    for column in log_2df:
        if not column == 'name' and not column == 'Season':
            log_2df[column] = pd.to_numeric(log_2df[column], errors='coerce')
    log_2df = log_2df.dropna()

    # filter out WS that were negative as the model needs positive valeus
    log_2df['WS'] = log_2df['WS'].map(lambda x: 0 if x < 0 else x)
    log_2df['DWS'] = log_2df['DWS'].map(lambda x: 0 if x < 0 else x)
    log_2df['OWS'] = log_2df['OWS'].map(lambda x: 0 if x < 0 else x)

    # define the player name
    player_name = ""
    for item in name.split():
        player_name += item + "-"

    player_name = player_name[:-1]

    player = []
    for index, row in log_2df.iterrows():
        if row[8] == player_name:
            if season in row[9]:
                player.append(row.tolist())
                # drop the player's season from the df
                # so it's not trained on it
                log_2df.drop(index, inplace=True)

    #print (player)

    arr = [1]
    for number in player[0][1:8]:
        arr.append(number)

    y, X = pa.dmatrices('MVP ~ ast_pct + Win_pct + DWS + OWS + reb_g'
                        ' + pts_g + WS'
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

    #print (result)

    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    #print (predicted)

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    #print (probs)

    # generate evaluation metrics
    #print (metrics.accuracy_score(y_test, predicted))
    print (model2.score(X, y))

    # print the name of the player and their chance of winning
    print (name + " had a %.2f" % (model.predict_proba(np.array(arr))[0][1]*100)
           + "% chance of winning MVP in the " + season +
           " season.")
    if player[0][0] == 1:
        print ("In the " + season + " season, " + name + " won MVP.")
    else:
        print ("In the " + season + " season, " + name + " didn't win MVP.")

def odds_of_mvp_advanced(name, season):
    # import the dataframe of players
    df = pd.read_csv('../basketball/mvp_totals.csv',
                     encoding='latin-1')

    # logistic regression using sklearn
    # First formalize the dataframe in a way that will work
    log_2df = df[['MVP', 'Win %', 'DWS', 'OWS', 'WS', 'name', 'Season']]

    log_2df['Win_pct'] = log_2df['Win %']

    log_2df = log_2df[['MVP', 'Win_pct', 'DWS', 'OWS',
                        'WS', 'name', 'Season']]

    # convert the columns to numeric
    for column in log_2df:
        if not column == 'name' and not column == 'Season':
            log_2df[column] = pd.to_numeric(log_2df[column], errors='coerce')
    log_2df = log_2df.dropna()

    # filter out WS that were negative as the model needs positive valeus
    log_2df['WS'] = log_2df['WS'].map(lambda x: 0 if x < 0 else x)
    log_2df['DWS'] = log_2df['DWS'].map(lambda x: 0 if x < 0 else x)
    log_2df['OWS'] = log_2df['OWS'].map(lambda x: 0 if x < 0 else x)

    # define the player name
    player_name = ""
    for item in name.split():
        player_name += item + "-"

    player_name = player_name[:-1]

    player = []
    for index, row in log_2df.iterrows():
        if row[5] == player_name:
            if season in row[6]:
                player.append(row.tolist())
                # drop the player's season from the df
                # so it's not trained on it
                log_2df.drop(index, inplace=True)

    #print(player)

    arr = [1]
    for number in player[0][1:5]:
        arr.append(number)

    y, X = pa.dmatrices('MVP ~ Win_pct + DWS + OWS '
                        ' + WS'
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

    result['p values'] = np.round(pvalues, 3)

    result['Coefficient'] = result['Coefficient'].map(lambda x: x[0])

    result.round({'Coefficient': 3})

    #print(result)

    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    # print (predicted)

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    # print (probs)

    # generate evaluation metrics
    # print (metrics.accuracy_score(y_test, predicted))
    print (model.score(X, y))

    # print the name of the player and their chance of winning
    print(name + " had a %.2f" % (model.predict_proba(np.array(arr))[0][1] * 100)
          + "% chance of winning MVP in the " + season +
          " season.")
    if player[0][0] == 1:
        print("In the " + season + " season, " + name + " won MVP.")
    else:
        print("In the " + season + " season, " + name + " didn't win MVP.")

# ignore the warnings because they work fine
warnings.filterwarnings("ignore")

# get the player and the year
name = input('Please enter a player name (full name with capital letters where applicable): ')
year = input('Please enter a season they played in (formatted 1990-91): ')
odds_of_mvp_with_normal_stats(name, year)
odds_of_mvp_advanced(name, year)
