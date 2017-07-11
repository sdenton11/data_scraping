from urllib.request import urlopen
from bs4 import BeautifulSoup
import pprint
import pandas as pd

url = 'http://www.basketball-reference.com/draft/NBA_2013.html'
html = urlopen(url)

soup = BeautifulSoup(html)

column_headers = [th.getText() for th
                  in soup.findAll('tr', limit = 2)[1].findAll('th')]

data_rows = soup.findAll('tr')[2:]

player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]

df = pd.DataFrame(player_data, columns=column_headers[1:])

cols = pd.Series(df.columns)

for dup in df.columns.get_duplicates():
    cols[df.columns.get_loc(dup)] = [dup + '/GAME' if d_idx == 1 else dup for d_idx in
                                     range(df.columns.get_loc(dup).sum())]

df.columns = cols

df['PTS/GAME'] = df['PTS/GAME'].map(lambda x: None if x == '' else x if x == None else float(x))

df['AST/GAME'] = df['AST/GAME'].map(lambda x: None if x == '' else x if x == None else float(x))

df['TRB/GAME'] = df['TRB/GAME'].map(lambda x: None if x == '' else x if x == None else float(x))

df['OFF/GAME'] = df['PTS/GAME'] + df['AST/GAME'] + df['TRB/GAME']

df = df.sort_values(by='OFF/GAME', ascending = False)

# get the column names and replace all '%' with '_Perc'
df.columns = df.columns.str.replace('%', '_Perc')

df.to_csv('2013_draft.csv')
