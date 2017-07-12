from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup



def file_maker(playerName):

    player = playerName.lower()

    def name_converter(string):
        name = string.split()
        return name[1][0] + "/" + name[1][:5] + name[0][:2] + "01.html"

    url = 'http://www.basketball-reference.com/players/' + name_converter(player)
    print (player)
    print(url)
    html = urlopen(url)

    soup = BeautifulSoup(html, "lxml")

    column_headers = [th.getText() for th
                      in soup.findAll('tr', limit = 2)[0].findAll('th')]

    data_rows = soup.findAll('tr')[1:]


    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]

    per_game_df = pd.DataFrame(player_data, columns=column_headers[1:])

    for column in per_game_df:
        per_game_df[column] = pd.to_numeric(per_game_df[column], errors='ignore')

    per_game_df['OFF'] = per_game_df['ORB'] + 2 * per_game_df['AST']\
                         + per_game_df['PTS'] - per_game_df['TOV']

    per_game_df['DEF'] = per_game_df['DRB'] + per_game_df['STL']\
                         + per_game_df['BLK']

    per_game_df['TOT'] = per_game_df['OFF'] + per_game_df['DEF']

    print (per_game_df)

    def file_name(string):
        names = string.split()
        return names[0] + "_" + names[1] + ".csv"

    per_game_df.to_csv(file_name(player))



"""
for player in team('http://www.basketball-reference.com/teams/PHO/2017.html'):
    file_maker(player)

"""
file_maker("Dell Curry")
