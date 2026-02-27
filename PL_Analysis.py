# IMporting necessary libraries for data analysis and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
pl_df = pd.read_csv('C:\\Users\\deep\\OneDrive\\Documents\\VS Code Projects\\PL Portfolio Project\\archive\\PremierLeague.csv')

# Display the first few rows of the dataset
pl_df.head()

# Check for missing values
print(pl_df.isnull().sum())

# Checking for number of games in each season. Should be 380 for each season, but we can see that there are some missing values.
pl_df['Season'].value_counts().sort_values()

# Checking for number of games played by each team. Should be 1216 for each team that were in the league for every season.
games_played_per_team = (
    pd.concat([pl_df['HomeTeam'], pl_df['AwayTeam']])
      .value_counts()
)
games_played_per_team.head(10)
\
# Checking for number of games played by each team in each season. Should be 38 for each team in each season, but we can see that there are some missing values.
games_per_season = (
    pd.concat([
        pl_df[['Season', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'}),
        pl_df[['Season', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'})
    ])
    .groupby(['Season', 'Team'])
    .size()
)
games_per_season

# Check for season with the most missing values
missing_values_per_season = pl_df.groupby('Season').apply(lambda x: x.isnull().sum())
print(missing_values_per_season)

## Creating a new DataFrame to calculate total points for each team in each season and then Ranking them by points.
team_points = pd.concat([
    pl_df[['Season', 'HomeTeam', 'HomeTeamPoints']].rename(columns={'HomeTeam': 'Team', 'HomeTeamPoints': 'Points'}),
    pl_df[['Season', 'AwayTeam', 'AwayTeamPoints']].rename(columns={'AwayTeam': 'Team', 'AwayTeamPoints': 'Points'})
])

# Group by Season and Team to sum the points
season_points = (
    team_points
    .groupby(['Season', 'Team'])
    .sum()
    .reset_index()
)

# Rank the teams by points within each season
season_points['Rank'] = season_points.groupby('Season')['Points'].rank(ascending=False, method='min').astype(int)

# Sort the DataFrame by Season and Rank
season_points = season_points.sort_values(by=['Season', 'Rank']).reset_index(drop=True)
season_points

# Calculate the squared difference between predicted rank (1) and actual rank
season_points['SquaredDiff'] = (1 - season_points['Rank']) ** 2

# Ensure the new column is placed right next to the 'Rank' column
# Get the position of the 'Rank' column
rank_col_idx = season_points.columns.get_loc('Rank')

# Insert the new column at the position right after 'Rank'
season_points.insert(rank_col_idx + 1, 'SquaredDiff', season_points.pop('SquaredDiff'))

# Display the updated DataFrame
season_points

# Calculate the Mean of the Squared Differences
mean_squared_error = season_points['SquaredDiff'].mean()

# 2. Take the Square Root
rmse = np.sqrt(mean_squared_error)
print(rmse)

# By using the Squared differences before averaging, your ranking accounts for volatility. 
# A team that finishes 1st and 5th will have a worse RMSE than a team that finishes 3rd and 3rd, even though their "average rank" (3.0) is the same. 
# RMSE "punishes" that one bad 5th-place finish more heavily.

# Group by Team and calculate the Mean of SquaredDiff
team_rmse = season_points.groupby('Team')['SquaredDiff'].mean().reset_index()

# Rename the column to MSE for clarity, then take the Square Root
team_rmse.rename(columns={'SquaredDiff': 'RMSE'}, inplace=True)
team_rmse['RMSE'] = np.sqrt(team_rmse['RMSE'])

# Sort by RMSE (Ascending = lowest error first)
team_rmse = team_rmse.sort_values(by='RMSE', ascending=True)

# Add a final Rank column for the teams themselves
team_rmse['Overall_Rank'] = range(1, len(team_rmse) + 1)

print(team_rmse)

# Visualizing the RMSE for each team with data labels for all teams.
plt.figure(figsize=(12, 6))
sns.barplot(x='Team', y='RMSE', data=team_rmse, palette='viridis')
plt.title('RMSE of Team Rankings Compared to 1st Place')
plt.xlabel('Team')
plt.ylabel('RMSE')
plt.ylim(0, 25.0)
plt.xticks(rotation=90)
# Adding data labels on top of each bar
ax = sns.barplot(x='Team', y='RMSE', data=team_rmse, palette='viridis')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, rotation=90)
plt.tight_layout()
plt.show()

###
# Now narrowing down on Arsenal's performance across the seasons. Particuarly interested in comparing the Arteta era to the Wenger era.

pl_df['Date'] = pd.to_datetime(pl_df['Date'])
pl_df.head()
arsenal_data = pl_df[(pl_df['HomeTeam'] == 'Arsenal') | (pl_df['AwayTeam'] == 'Arsenal')]
arsenal_arteta = arsenal_data[arsenal_data['Date'] >= pd.to_datetime('2019-12-20')]
arsenal_arteta.to_csv('arsenal_data_since_2019.csv', index=False)
arsenal_arteta.head()
arsenal_wenger = arsenal_data[(arsenal_data['Date'] < pd.to_datetime('2018-07-01')) & (arsenal_data['Date'] >= pd.to_datetime('1996-10-01'))]
arsenal_wenger.to_csv('arsenal_data_before_2018.csv', index=False)
arsenal_wenger.head()

# Arteta Era: 2019 onwards
arteta_home_games = arsenal_arteta[arsenal_arteta['HomeTeam'] == 'Arsenal']
arteta_away_games = arsenal_arteta[arsenal_arteta['AwayTeam'] == 'Arsenal']
# Calculate wins, draws, and losses using arsenal_data_since_2019
a_home_wins = len(arteta_home_games[arteta_home_games['FullTimeResult'] == 'H'])
a_away_wins = len(arteta_away_games[arteta_away_games['FullTimeResult'] == 'A'])
a_home_draws = len(arteta_home_games[arteta_home_games['FullTimeResult'] == 'D'])
a_away_draws = len(arteta_away_games[arteta_away_games['FullTimeResult'] == 'D'])
a_home_losses = len(arteta_home_games[arteta_home_games['FullTimeResult'] == 'A'])
a_away_losses = len(arteta_away_games[arteta_away_games['FullTimeResult'] == 'H'])

# Print the results
print("Total Number of Home Wins:", a_home_wins)
print("Total Number of Away Wins:", a_away_wins)
print("Total Number of Home Draws:", a_home_draws)
print("Total Number of Away Draws:", a_away_draws)
print("Total Number of Home Losses:", a_home_losses)
print("Total Number of Away Losses:", a_away_losses)

# Data for the bar chart
labels = ['Home Wins', 'Away Wins', 'Home Draws', 'Away Draws', 'Home Losses', 'Away Losses']
values = [a_home_wins, a_away_wins, a_home_draws, a_away_draws, a_home_losses, a_away_losses]

# Create the bar chart
plt.bar(labels, values)
plt.xlabel('Result')
plt.ylabel('Count')
plt.title('Arsenal Wins, Draws, and Losses After December 20, 2019')
plt.xticks(rotation=45, ha='right')
plt.show()

# Calculate total goals scored and conceded
a_hg_scored = arsenal_arteta[arsenal_arteta['HomeTeam'] == 'Arsenal']['FullTimeHomeTeamGoals'].sum()
a_ag_scored = arsenal_arteta[arsenal_arteta['AwayTeam'] == 'Arsenal']['FullTimeAwayTeamGoals'].sum()
a_hg_conceded = arsenal_arteta[arsenal_arteta['HomeTeam'] == 'Arsenal']['FullTimeAwayTeamGoals'].sum()
a_ag_conceded = arsenal_arteta[arsenal_arteta['AwayTeam'] == 'Arsenal']['FullTimeHomeTeamGoals'].sum()

# Data for the bar chart
labels = ['Home Scored', 'Away Scored', 'Home Conceded', 'Away Conceded']
values = [a_hg_scored, a_ag_scored, a_hg_conceded, a_ag_conceded]

# Create the bar chart
plt.bar(labels, values)
plt.xlabel('Goals')
plt.ylabel('Count')
plt.title('Arsenal Goals Scored and Conceded (Home and Away)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Wenger Era: 1996-2018
wenger_home_games = arsenal_wenger[arsenal_wenger['HomeTeam'] == 'Arsenal']
wenger_away_games = arsenal_wenger[arsenal_wenger['AwayTeam'] == 'Arsenal']
# Calculate wins, draws, and losses using arsenal_data_before_2018    
w_home_wins = len(wenger_home_games[wenger_home_games['FullTimeResult'] == 'H'])
w_away_wins = len(wenger_away_games[wenger_away_games['FullTimeResult'] == 'A'])
w_home_draws = len(wenger_home_games[wenger_home_games['FullTimeResult'] == 'D'])
w_away_draws = len(wenger_away_games[wenger_away_games['FullTimeResult'] == 'D'])
w_home_losses = len(wenger_home_games[wenger_home_games['FullTimeResult'] == 'A'])
w_away_losses = len(wenger_away_games[wenger_away_games['FullTimeResult'] == 'H'])

# Print the results
print("Total Number of Home Wins:", w_home_wins)
print("Total Number of Away Wins:", w_away_wins)
print("Total Number of Home Draws:", w_home_draws)
print("Total Number of Away Draws:", w_away_draws)
print("Total Number of Home Losses:", w_home_losses)
print("Total Number of Away Losses:", w_away_losses)

# Data for the bar chart
labels = ['Home Wins', 'Away Wins', 'Home Draws', 'Away Draws', 'Home Losses', 'Away Losses']
values = [w_home_wins, w_away_wins, w_home_draws, w_away_draws, w_home_losses, w_away_losses]

# Create the bar chart
plt.bar(labels, values)
plt.xlabel('Result')
plt.ylabel('Count')
plt.title('Arsenal Wins, Draws, and Losses from 1996 to 2018')
plt.xticks(rotation=45, ha='right')
plt.show()

# Calculate total goals scored and conceded
w_hg_scored = arsenal_wenger[arsenal_wenger['HomeTeam'] == 'Arsenal']['FullTimeHomeTeamGoals'].sum()
w_ag_scored = arsenal_wenger[arsenal_wenger['AwayTeam'] == 'Arsenal']['FullTimeAwayTeamGoals'].sum()
w_hg_conceded = arsenal_wenger[arsenal_wenger['HomeTeam'] == 'Arsenal']['FullTimeAwayTeamGoals'].sum()
w_ag_conceded = arsenal_wenger[arsenal_wenger['AwayTeam'] == 'Arsenal']['FullTimeHomeTeamGoals'].sum()

# Data for the bar chart
labels = ['Home Scored', 'Away Scored', 'Home Conceded', 'Away Conceded']
values = [w_hg_scored, w_ag_scored, w_hg_conceded, w_ag_conceded]

# Create the bar chart
plt.bar(labels, values)
plt.xlabel('Goals')
plt.ylabel('Count')
plt.title('Arsenal Goals Scored and Conceded (Home and Away) from 1996 to 2018')
plt.xticks(rotation=45, ha='right')
plt.show()

####
# Compare both eras side by side for win percentage, goals scored per game, goals conceded per game, and goal difference per game.

# Calculate total games played in each era.
arteta_games = len(arsenal_arteta)
arteta_games
wenger_games = len(arsenal_wenger)
wenger_games

# Calculate win percentages for each era
arteta_wins = a_home_wins + a_away_wins
wenger_wins = w_home_wins + w_away_wins

arteta_win_percentage = ((arteta_wins / arteta_games) * 100)
wenger_win_percentage = ((wenger_wins / wenger_games) * 100) 

# Calculate goals scored and conceded per game for each era
arteta_goals_scored_per_game = (a_hg_scored + a_ag_scored) / arteta_games 
arteta_goals_conceded_per_game = (a_hg_conceded + a_ag_conceded) / arteta_games 

wenger_goals_scored_per_game = (w_hg_scored + w_ag_scored) / wenger_games 
wenger_goals_conceded_per_game = (w_hg_conceded + w_ag_conceded) / wenger_games 

# Calculate goal difference per game for each era
arteta_goal_difference_per_game = (arteta_goals_scored_per_game - arteta_goals_conceded_per_game)
wenger_goal_difference_per_game = (wenger_goals_scored_per_game - wenger_goals_conceded_per_game)

# Print the results
print("Arteta Era Games:", arteta_games)
print("Wenger Era Games:", wenger_games)
print("Arteta Win Percentage:", arteta_win_percentage)
print("Wenger Win Percentage:", wenger_win_percentage)
print("Arteta Goals Scored Per Game:", arteta_goals_scored_per_game)
print("Wenger Goals Scored Per Game:", wenger_goals_scored_per_game)
print("Arteta Goals Conceded Per Game:", arteta_goals_conceded_per_game)
print("Wenger Goals Conceded Per Game:", wenger_goals_conceded_per_game)
print("Arteta Goal Difference Per Game:", arteta_goal_difference_per_game)
print("Wenger Goal Difference Per Game:", wenger_goal_difference_per_game)

# Visualize the comparison between the two eras
labels = ['Win Percentage', 'Goals Scored Per Game', 'Goals Conceded Per Game', 'Goal Difference Per Game']
arteta_values = [arteta_win_percentage, arteta_goals_scored_per_game, arteta_goals_conceded_per_game, arteta_goal_difference_per_game]
wenger_values = [wenger_win_percentage, wenger_goals_scored_per_game, wenger_goals_conceded_per_game, wenger_goal_difference_per_game]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, arteta_values, width, label='Arteta Era')
rects2 = ax.bar(x + width/2, wenger_values, width, label='Wenger Era')
ax.set_ylabel('Values')
ax.set_title('Comparison of Arteta Era vs Wenger Era')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# Adding data labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

###
# Narrowing down on Arsenal's performance against their top 6 rivals in the Arteta era and comparing it to the Wenger era. The top 6 rivals are: Manchester United, Manchester City, Chelsea, Liverpool, Tottenham, and Arsenal.

# Create a list of top 6 rivals
top_6_rivals = ['Man United', 'Man City', 'Chelsea', 'Liverpool', 'Tottenham']
# Filter the Arsenal data for matches against top 6 rivals in the Arteta era
arteta_top_6 = arsenal_arteta[(arsenal_arteta['HomeTeam'].isin(top_6_rivals)) | (arsenal_arteta['AwayTeam'].isin(top_6_rivals))]
arteta_top_6.head()
# Filter the Arsenal data for matches against top 6 rivals in the Wenger era    
wenger_top_6 = arsenal_wenger[(arsenal_wenger['HomeTeam'].isin(top_6_rivals)) | (arsenal_wenger['AwayTeam'].isin(top_6_rivals))]
wenger_top_6.head()

# Calculate win percentages for matches against top 6 rivals in each era
arteta_top_6_wins = len(arteta_top_6[(arteta_top_6['HomeTeam'] == 'Arsenal') & (arteta_top_6['FullTimeResult'] == 'H')]) + len(arteta_top_6[(arteta_top_6['AwayTeam'] == 'Arsenal') & (arteta_top_6['FullTimeResult'] == 'A')])
wenger_top_6_wins = len(wenger_top_6[(wenger_top_6['HomeTeam'] == 'Arsenal') & (wenger_top_6['FullTimeResult'] == 'H')]) + len(wenger_top_6[(wenger_top_6['AwayTeam'] == 'Arsenal') & (wenger_top_6['FullTimeResult'] == 'A')])
arteta_top_6_games = len(arteta_top_6)
arteta_top_6_games
wenger_top_6_games = len(wenger_top_6)
wenger_top_6_games
arteta_top_6_win_percentage = ((arteta_top_6_wins / arteta_top_6_games) * 100)
wenger_top_6_win_percentage = ((wenger_top_6_wins / wenger_top_6_games) * 100)
# Print the results
print("Arteta Era Games Against Top 6 Rivals:", arteta_top_6_games)
print("Wenger Era Games Against Top 6 Rivals:", wenger_top_6_games)
print("Arteta Win Percentage Against Top 6 Rivals:", arteta_top_6_win_percentage)
print("Wenger Win Percentage Against Top 6 Rivals:", wenger_top_6_win_percentage)

# Combine the results into a DataFrame for visualization
comparison_df = pd.DataFrame({
    'Era': ['Arteta Era', 'Wenger Era'],
    'Win Percentage Against Top 6 Rivals': [arteta_top_6_win_percentage, wenger_top_6_win_percentage]
})
# Visualize the comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='Era', y='Win Percentage Against Top 6 Rivals', data=comparison_df, palette='viridis')
plt.title('Arsenal Win Percentage Against Top 6 Rivals: Arteta Era vs Wenger Era')
plt.ylabel('Win Percentage (%)')
plt.ylim(0, 100)
# Adding data labels on top of each bar
ax = sns.barplot(x='Era', y='Win Percentage Against Top 6 Rivals', data=comparison_df, palette='viridis')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', padding=3)
plt.tight_layout()
plt.show()

###
# Now comparing Arsenal's performance against each of the top 6 rivals individually in the Arteta era and the Wenger era. 
# This will give us a more granular view of how Arsenal has performed against each rival under both managers.

# Initialize a DataFrame to store the results
results_list = []

# Loop through each rival and calculate win percentages for both eras
for rival in top_6_rivals:
    
    # Calculate win percentage for Arteta era
    arteta_rival = arteta_top_6[(arteta_top_6['HomeTeam'] == rival) | (arteta_top_6['AwayTeam'] == rival)]
    arteta_wins = len(arteta_rival[(arteta_rival['HomeTeam'] == 'Arsenal') & (arteta_rival['FullTimeResult'] == 'H')]) + \
                  len(arteta_rival[(arteta_rival['AwayTeam'] == 'Arsenal') & (arteta_rival['FullTimeResult'] == 'A')])
    arteta_games = len(arteta_rival)
    
    if arteta_games > 0:
        arteta_win_percentage = (arteta_wins / arteta_games) * 100
    else:
        arteta_win_percentage = 0
        
    # Calculate win percentage for Wenger era
    wenger_rival = wenger_top_6[(wenger_top_6['HomeTeam'] == rival) | (wenger_top_6['AwayTeam'] == rival)]
    wenger_wins = len(wenger_rival[(wenger_rival['HomeTeam'] == 'Arsenal') & (wenger_rival['FullTimeResult'] == 'H')]) + \
                  len(wenger_rival[(wenger_rival['AwayTeam'] == 'Arsenal') & (wenger_rival['FullTimeResult'] == 'A')])
    wenger_games = len(wenger_rival)
    
    if wenger_games > 0:
        wenger_win_percentage = (wenger_wins / wenger_games) * 100
    else:
        wenger_win_percentage = 0
        
    # 2. Append the dictionary to your standard Python list
    results_list.append({
        'Rival': rival,
        'Arteta Win Percentage': arteta_win_percentage,
        'Wenger Win Percentage': wenger_win_percentage
    })

# 3. Convert the populated list into a DataFrame all at once!
results = pd.DataFrame(results_list)
# Check out the final results
print(results)

# Visualize the comparison for each rival
results_melted = results.melt(id_vars='Rival', var_name='Era', value_name='Win Percentage')
plt.figure(figsize=(10, 6))
# Draw the grouped barplot and assign it to 'ax'
ax = sns.barplot(x='Rival', y='Win Percentage', hue='Era', data=results_melted, palette='viridis')
plt.title('Arsenal Win Percentage Against Each Top 6 Rival: Arteta Era vs Wenger Era')
plt.ylabel('Win Percentage (%)')
plt.ylim(0, 100)
# Add data labels to each individual bar in the grouped chart
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.show()

# This analysis provides a detailed comparison of Arsenal's performance under Arteta and Wenger, both overall and against their top 6 rivals. 
# It highlights areas of strength and weakness for each era, offering insights into how the team has evolved over time.