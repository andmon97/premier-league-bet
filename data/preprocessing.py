import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

DATASET_FOLDER = 'raw'
FILENAME = 'matches.csv'

file_path = f'{DATASET_FOLDER}/{FILENAME}'
df = pd.read_csv(file_path)



# looking at the data descriptions notebook, we remove the useless columns and with nan
df.drop(columns=["Unnamed: 0", "comp", "round", "attendance", "notes", "match report"],inplace=True);
#print(df.head())



# cast dataset columns in correct type
df["date"] = pd.to_datetime(df["date"])
df["venue"] = df["venue"].astype("category")
df["opponent"] = df["opponent"].astype("category")
df["team"] = df["team"].astype("category")
df["result"] = df["result"].astype("category")
df["day"] = df["day"].astype("category")



df["hour"] = df["time"].str.replace(":.+", "", regex=True).astype("int")
df["day_code"] = df["date"].dt.day_of_week



# clean formation column
df.formation = df.formation.str.replace("◆", "")
df.formation = df.formation.str.replace("-0", "")
# collaps formation less present (<100) in "other" category
formation_value_counts = df.formation.value_counts()
formation_to_replace = formation_value_counts[formation_value_counts < 100].index
df["formation"] = df["formation"].replace(formation_to_replace, 'Other')



# print if there are duplicates for each season
print(df["season"].value_counts())
df_cols = df.columns
df_cols = df_cols.drop("season")

print(df.duplicated(subset=df_cols).sum())
df.drop_duplicates(subset=df_cols, inplace=True)
print(df.duplicated(subset=df_cols).sum())

def get_correct_season(date):
    if pd.isna(date):
        return None  # Gestisci i valori NaN
    date = pd.to_datetime(date, errors='coerce')  # Converti con 'coerce' per gestire errori
    if pd.isna(date):
        return None  # Se la conversione fallisce, restituisci None
    if date.month >= 8:
        return date.year + 1
    else:
        return date.year
        
df["season"] = df["date"].apply(get_correct_season)
# print the number of matches for each season, sorted by season
#print(df["season"].value_counts().sort_index())

df["points"] = df["result"].apply(lambda x: 3 if x == "W" else 1 if x == "D" else 0)
df["points"] = df["points"].astype("int")
# get the season winners
# This is a multi-step operation to get the winner of each season:
# 1. group by season and team, and sum the points for each team in each season
# 2. sort the results by season and points in descending order
# 3. group by season and get the team with the highest points for each season
# 4. reset the index to get a dataframe with season and team columns
# 5. sort the results by season
# The result is a dataframe with the season and the winner of that season.
season_winners = df.groupby(["season", "team"], observed=False)["points"].sum().reset_index() \
    .sort_values(["season", "points"], ascending=[True, False]) \
    .groupby("season", observed=False).first()

print(season_winners)
df['season_winner'] = df['season'].map(season_winners['team'])



def captains_func(data):
    if data['count'] == 0:
        data['count'] = np.nan
    return data
# Group the DataFrame by 'team' and count occurrences of each 'captain' within each group
group = df.groupby('team', observed=False)['captain'].value_counts().reset_index(name='count')
# Apply the 'captains_func' to each row of the group DataFrame
# This function replaces 'count' with NaN if the count is 0
group = group.apply(captains_func, axis=1)
# Remove rows from the group DataFrame where 'count' is NaN
group.dropna(inplace=True)
# Drop the 'count' column from the group DataFrame, leaving only 'team' and 'captain'
group = group.drop(columns='count')
group['team'].value_counts()    
group[group['team'] == 'Liverpool']

df['date'] = pd.to_datetime(df['date'])

# Sort the DataFrame by team and date
df_sorted = df.sort_values(['team', 'date'])

# Reset the index to reflect the new order
df_sorted = df_sorted.reset_index(drop=True)

# Verify the sorting
def verify_sorting(data):
    # Check if dates are in ascending order for each team
    is_sorted = data.groupby('team', observed=False)['date'].is_monotonic_increasing.all()
    
    if is_sorted:
        print("Data is correctly sorted by date for each team.")
    else:
        print("WARNING: Data is not correctly sorted. Please check for inconsistencies.")

# Run the verification
verify_sorting(df_sorted)

# numeric features
num_cols = ['sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'xga', 'xg', 'gf', 'ga']
for col in num_cols:
    df_sorted[col] = pd.to_numeric(df_sorted[col])
# use to calculate ratios for free kick and penalty kicks
def calculate_fk_pk_ratios(data):

    data['fk_ratio'] = data['fk'] / data['sh']
    
    data['pk_conversion_rate'] = data['pk'] / data['pkatt']
    
    data['pk_per_shot'] = data['pkatt'] / data['sh']
    
    data['fk_ratio'] = data['fk_ratio'].replace([np.inf, -np.inf], np.nan)
    data['pk_conversion_rate'] = data['pk_conversion_rate'].replace([np.inf, -np.inf], np.nan)
    data['pk_per_shot'] = data['pk_per_shot'].replace([np.inf, -np.inf], np.nan)
    
    data['fk_percentage'] = data['fk_ratio'] * 100
    data['pk_conversion_percentage'] = data['pk_conversion_rate'] * 100
    data['pk_per_shot_percentage'] = data['pk_per_shot'] * 100
    
    return data
# apply the ratios on the sorted dataframe and visualize
df_sorted = calculate_fk_pk_ratios(df_sorted)
df_sorted.drop(['pk_conversion_rate', 'pk_conversion_percentage'], axis=1, inplace=True)
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
i = 0
for col in ['fk_ratio', 'pk_per_shot', 'fk_percentage', 'pk_per_shot_percentage']:
    sns.histplot(df_sorted[col], kde=True, ax=axs.flatten()[i])
    axs.flatten()[i].set_title('Distribution of ' + col)
    i += 1

plt.tight_layout()
plt.show()
# print stats for ratios
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
i = 0
for col in ['fk_ratio', 'pk_per_shot', 'fk_percentage', 'pk_per_shot_percentage']:
    sns.boxplot(x=df_sorted[col], ax=axs.flatten()[i])  # Changed to boxplot
    axs.flatten()[i].set_title('Distribution of ' + col)
    i += 1
print(df_sorted[['fk_ratio', 'pk_per_shot', 'fk_percentage', 'pk_per_shot_percentage']].agg(['mean', 'min', 'max']))
plt.tight_layout()
plt.show()



# compute rolling average for each team of numeric stats
def calculate_rolling_average(data, column, window=5):
    """
    Calculate the rolling average of a column for each team.
    
    Parameters:
    data (DataFrame): The input DataFrame
    column (str): The column to calculate the rolling average for
    window (int): The number of games to include in the rolling average
    
    Returns:
    Series: The rolling average
    """
    return data.groupby('team', observed=False)[column].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

df_sorted['rolling_xg'] = calculate_rolling_average(df_sorted, 'xg')
df_sorted['rolling_xga'] = calculate_rolling_average(df_sorted, 'xga')
df_sorted['rolling_poss'] = calculate_rolling_average(df_sorted, 'poss')
df_sorted['rolling_sh'] = calculate_rolling_average(df_sorted, 'sh')
df_sorted['rolling_sot'] = calculate_rolling_average(df_sorted, 'sot')
df_sorted['rolling_dist'] = calculate_rolling_average(df_sorted, 'dist')

df_sorted['result_encoded'] = pd.to_numeric(df_sorted['result'].map({'W': 1, 'D': 0, 'L': -1}))
df_sorted['form'] = calculate_rolling_average(df_sorted, 'result_encoded')
df_sorted['goal_diff'] = df_sorted['gf'] - df_sorted['ga']
df_sorted['rolling_goal_diff'] = calculate_rolling_average(df_sorted, 'goal_diff')