import pandas as pd

#%%
# Using the SQL query from Survival MDN paper
df_full = pd.read_json("data/MIMIC/MIMIC_IV_hosp_failure.json", lines=True)
df_full = df_full.drop_duplicates(subset=['subject_id', 'hadm_id'], keep='last')
df_full.reset_index(drop=True, inplace=True)
insta_nan_death_time = df_full.time_to_death.isna()
df_full.loc[insta_nan_death_time, 'time_to_death'] = df_full.loc[insta_nan_death_time, 'los_hosp']
# Whether drop the patient with negative or zero time_to_death
# insta_non_pos_time = df_full.time_to_death < 0.5
# df_full.loc[insta_non_pos_time, 'time_to_death'] = df_full.loc[insta_non_pos_time, 'los_icu_days']
# df_full.loc[insta_non_pos_time, 'death'] = 0
df_full.time_to_death = df_full.time_to_death.round().astype(int)
df_full = df_full.rename(columns={'time_to_death': 'time'})
df_full = df_full.drop(columns=['subject_id', 'hadm_id', 'los_hosp'])

#%%
# Drop people with negative or zero times
insta_with_non_pos_time = df_full.time < 1
df_full = df_full[~insta_with_non_pos_time]
df_full.reset_index(drop=True, inplace=True)

df_full.to_csv("data/MIMIC/MIMIC_IV_hosp_failure.csv", index=False)