import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pprint import pprint

# URL of the Premier League player stats page on FBref
url = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'


# Function to get the page content
def get_page_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    return response.content


# Function to parse the player data
def parse_player_data(content):
    soup = BeautifulSoup(content, 'html.parser')

    # Print the content for debugging purposes
    with open('fbref_page.html', 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    player_table = soup.find('table', {'id': 'stats_standard'})
    if player_table is None:
        print("Player table not found.")
        return []

    player_rows = player_table.find('tbody').find_all('tr')

    players = []
    for row in player_rows:
        player = {}
        columns = row.find_all('td')

        if columns:
            player['name'] = columns[0].get_text(strip=True)
            player['profile_url'] = 'https://fbref.com' + columns[0].find('a')['href']
            players.append(player)

    return players


# Function to parse local HTML file
def parse_local_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
    return soup

# File path to your local HTML file
file_path = 'fbref_page.html'



soup = parse_local_html(file_path)

def get_player_list(soup):
    # Parse the local HTML file
    player_table = soup.find('table', {'id': 'stats_standard'})
    player_rows = player_table.find('tbody').find_all('tr')

    players = []
    pprint(player_rows[0])
    for row in player_rows:
        player = {}
        columns = row.find_all('td')

        if columns:
            player['name'] = columns[0].get_text(strip=True)
            player['nationality'] = columns[1].get_text(strip=True)
            player['position'] = columns[2].get_text(strip=True)
            player['team'] = columns[3].get_text(strip=True)
            player['age'] = columns[4].get_text(strip=True)
            player['matches_played'] = columns[6].get_text(strip=True)
            player['starts'] = columns[7].get_text(strip=True)
            player['minutes'] = columns[8].get_text(strip=True)
            player['goals'] = columns[10].get_text(strip=True)
            player['assists'] = columns[11].get_text(strip=True)
            player['non_penalty_goals'] = columns[13].get_text(strip=True)
            player['penalty_Kicks_made'] = columns[14].get_text(strip=True)
            player['yellow_cards'] = columns[16].get_text(strip=True)
            player['red_cards'] = columns[17].get_text(strip=True)
            player['xg'] = columns[18].get_text(strip=True)
            player['non_penalty_xg'] = columns[19].get_text(strip=True)
            player['x_assisted_goals'] = columns[20].get_text(strip=True)
            player['progressive_carries'] = columns[22].get_text(strip=True)
            player['progressive_passes'] = columns[23].get_text(strip=True)
            player['progressive_passes_received'] = columns[24].get_text(strip=True)

            player['profile_url'] = 'https://fbref.com' + columns[0].find('a')['href']
            players.append(player)

    return players

df = pd.DataFrame(get_player_list(soup))


# Function to calculate similarity using Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Function to find similar players to a query player
def find_similar_players(df, query_player, top_n=10):
    # Exclude the query player from the dataset
    data_for_similarity = df[df['name'] != query_player].set_index('name')

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = ['non_penalty_goals', 'goals', 'assists', 'xg', 'non_penalty_xg', 'x_assisted_goals','progressive_carries','progressive_passes','progressive_passes_received']
    data_for_similarity[numerical_features] = scaler.fit_transform(data_for_similarity[numerical_features])

    # Calculate similarity with the query player
    query_stats = df[df['name'] == query_player][numerical_features].values.flatten().astype(float)
    data_for_similarity['similarity'] = data_for_similarity.apply(
        lambda row: euclidean_distance(query_stats.astype(float), row[numerical_features].values.flatten().astype(float)),
        axis=1
    )

    # Sort by similarity and select top N similar players
    similar_players = data_for_similarity.sort_values(by='similarity').head(top_n)

    return similar_players.reset_index()


# Example: Find 5 players similar to 'Player A'
query_player = 'Bruno Fernandes'

similar_players = find_similar_players(df, query_player)
print(f"Players similar to '{query_player}':")
print(similar_players[['name', 'similarity']])