import pandas as pd
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
print("pito")
# print the number of matches for each season, sorted by season
print(df["season"].value_counts().sort_index())

