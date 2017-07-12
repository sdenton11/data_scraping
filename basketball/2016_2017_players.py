import glob
import pandas as pd
import numpy as np

paths = ["./teams/east/atlanta_hawks/*.csv",
         "./teams/east/boston_celtics/*.csv",
         "./teams/east/brooklyn_nets/*.csv",
         "./teams/east/charlotte_hornets/*.csv",
         "./teams/east/chicago_bulls/*.csv",
         "./teams/east/cleveland_cavaliers/*.csv",
         "./teams/east/detroit_pistons/*.csv",
         "./teams/east/indiana_pacers/*.csv",
         "./teams/east/miami_heat/*.csv",
         "./teams/east/milwaukee_bucks/*.csv",
         "./teams/east/new_york_knicks/*.csv",
         "./teams/east/orlando_magic/*.csv",
         "./teams/east/philadelphia_76ers/*.csv",
         "./teams/east/toronto_raptors/*.csv",
         "./teams/east/washington_wizards/*.csv",
         "./teams/west/dallas_mavericks/*.csv",
         "./teams/west/denver_nuggets/*.csv",
         "./teams/west/golden_state_warriors/*.csv",
         "./teams/west/houston_rockets/*.csv",
         "./teams/west/los_angeles_clippers/*.csv",
         "./teams/west/los_angeles_lakers/*.csv",
         "./teams/west/memphis_grizzlies/*.csv",
         "./teams/west/minnesota_timberwolves/*.csv",
         "./teams/west/new_orleans_pelicans/*.csv",
         "./teams/west/oklahoma_city_thunder/*.csv",
         "./teams/west/phoenix_suns/*.csv",
         "./teams/west/portland_trail_blazers/*.csv",
         "./teams/west/sacramento_kings/*.csv",
         "./teams/west/san_antonio_spurs/*.csv",
         "./teams/west/utah_jazz/*.csv"]


player_list = []

for path in paths:
    for fname in glob.glob(path):
        df = pd.read_csv(fname, encoding='latin-1')
        df = df.dropna(subset=['Pos'])
        df = df.tail(1)
        df['name'] = fname.split("/")[len(fname.split("/")) - 1].split(".")[0]
        player_list.append(df)

player_df = pd.concat(player_list)
player_df = player_df.drop_duplicates()
player_df = player_df[player_df['MP'] > 20]
player_df['AST/TOV'] = player_df['AST']/player_df['TOV']
player_df = player_df.sort("AST/TOV", ascending=False)

player_df.to_csv("players.csv")