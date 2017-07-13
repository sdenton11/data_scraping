from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup
import re

def get_advanced_stats(link):
    url = link
    print(url)
    html = urlopen(url)

    soup = BeautifulSoup(html, "lxml")


    tables = soup.findAll('table', attrs={"class":"tablesaw compact"})

    table = tables[3]

    cols = [th.getText() for th in table.findAll('th')]

    data_rows = [table for table in table.findAll('tr')]

    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                    for i in range(len(data_rows))]


    player_data = player_data[1:]

    advanced_df = pd.DataFrame(player_data, columns=cols)

    for column in advanced_df:
        print (column)
        advanced_df[column] = pd.to_numeric(advanced_df[column], errors='ignore')

        return (advanced_df)

player = "/player/Chris-Andersen/Summary/866"
website = "http://basketball.realgm.com/" + player

print (get_advanced_stats(website))