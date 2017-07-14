from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup

def get_full_stats(link):
    url = "http://basketball.realgm.com/" + link
    print(url)
    html = urlopen(url)

    soup = BeautifulSoup(html, "lxml")


    tables = soup.findAll('table', attrs={"class":"tablesaw compact"})

    def create_df(chart):

        cols = [th.getText() for th in chart.findAll('th')]

        data_rows = [chart for chart in chart.findAll('tr')]

        player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                        for i in range(len(data_rows))]


        player_data = player_data[1:]

        advanced_df = pd.DataFrame(player_data, columns=cols)

        for column in advanced_df:
            advanced_df[column] = pd.to_numeric(advanced_df[column], errors='ignore')

        return (advanced_df)

    # assign the first table as the full table
    full_df = create_df(tables[0])

    # rename the necessary columns of the first table
    full_df = full_df.rename(index=str, columns={"MIN": "MIN/G",
                                                 "FGA": "FGA/G",
                                                 "FGM": "FGM/G",
                                                 "3PM": "3PM/G",
                                                 "3PA": "3PA/G",
                                                 "FTM": "FTM/G",
                                                 "FTA": "FTA/G",
                                                 "OFF": "ORB/G",
                                                 "DEF": "DRB/G",
                                                 "TRB": "TRB/G",
                                                 "AST": "AST/G",
                                                 "STL": "STL/G",
                                                 "BLK": "BLK/G",
                                                 "PF": "PF/G",
                                                 "TOV": "TOV/G",
                                                 "PTS": "PTS/G"})

    #iterate through the rest of the tables to add and merge them
    for table in tables[1:4]:
        full_df = pd.merge(full_df, create_df(table), on='Season', how='outer')


    full_df = full_df.T.drop_duplicates().T

    # rename the columns that are weird from the merge
    full_df = full_df.rename(index=str, columns={"Team_x": "Team",
                                                 "GP_x": "GP",
                                                 "GS_x": "GS",
                                                 "FG%_x": "FG%",
                                                 "3P%_x": "3P%",
                                                 "FT%_x": "FT%"})

    return (full_df)

# iterate through each player in the list of players to get their links
html = urlopen("http://basketball.realgm.com/nba/players/2017")
soup = BeautifulSoup(html, "lxml")

player_names = soup.findAll('table')

players = [player for player in player_names[0].findAll('tr')]
print (players)


links = [[a['href'] for a in players[i].findAll('a', href=True)]
         for i in range(len(players))]

# iterate throuch each link in the links
for link in links[1:]:
    direct = link[0]

    df = get_full_stats(direct)
    df.to_csv('./players_advanced_stats/%s.csv'%direct.split("/")[2])