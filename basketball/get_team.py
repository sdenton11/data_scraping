from urllib.request import urlopen
from bs4 import BeautifulSoup
import pprint
import pandas as pd

def team(name):
    url = name
    html = urlopen(url)

    soup = BeautifulSoup(html)

    column_headers = [th.getText() for th
                      in soup.findAll('tr', limit = 2)[0].findAll('th')]

    data_rows = soup.findAll('tr')[1:]

    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]


    df = pd.DataFrame(player_data, columns=column_headers[1:])

    return (df['Player'].tolist())

team('http://www.basketball-reference.com/teams/BOS/2017.html')