from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup

url = 'http://www.nba.com/summerleague/2017/stats#vegas'
print(url)
html = urlopen(url)

soup = BeautifulSoup(html, "lxml")

print (soup.findAll('tr')[15])

column_headers = [th.getText() for th
                  in soup.findAll('tr')[15].findAll('th')]

data_rows = soup.findAll('tr')[16:]

print(column_headers)
print (data_rows)

player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

per_game_df = pd.DataFrame(player_data, columns=column_headers)

for column in per_game_df:
    print (column)
    if (not column == 'PLAYER' and not column == 'TEAM'):
        per_game_df[column] = pd.to_numeric(per_game_df[column], errors='coerce')

per_game_df['VALUE'] = per_game_df['PTS'] + \
                       per_game_df['RPG'] + per_game_df ['APG'] +\
                       per_game_df['SPG'] + per_game_df['BPG']

print (per_game_df.dtypes)

per_game_df = per_game_df.sort("VALUE", ascending=False)

per_game_df.to_csv("vegas_summer_league.csv")