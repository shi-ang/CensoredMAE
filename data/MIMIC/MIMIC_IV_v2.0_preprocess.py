import pandas as pd

#%%
# Using the SQL query from Survival MDN paper
df_full = pd.read_json("data/MIMIC/MIMIC_IV_v2.0_SMDN.json", lines=True)
df_full = df_full.sort_values(by=['subject_id', 'hadm_id', 'stay_id'])
df_full = df_full.drop_duplicates(subset=['subject_id'], keep='last')
insta_nan_death_time = df_full.time_to_death.isna()
df_full.loc[insta_nan_death_time, 'time_to_death'] = df_full.loc[insta_nan_death_time, 'los_icu_days']
# Whether drop the patient with negative or zero time_to_death
# insta_non_pos_time = df_full.time_to_death < 0.5
# df_full.loc[insta_non_pos_time, 'time_to_death'] = df_full.loc[insta_non_pos_time, 'los_icu_days']
# df_full.loc[insta_non_pos_time, 'death'] = 0
df_full.time_to_death = df_full.time_to_death.round().astype(int)
df_full = df_full.rename(columns={'time_to_death': 'time', 'death': 'event',
                                  'glucose_min_1': 'glucose_1_min', 'glucose_max_1': 'glucose_1_max'})
df_full = df_full.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'los_icu_days', 'los_hosp_days',
                                'subject_id_1', 'stay_id_1', 'subject_id_2', 'stay_id_2', 'subject_id_3',
                                'stay_id_3', 'hadm_id_1'])

#%%
# Drop features with too many missing values
feature_threshold = 0.4
columns_need_attention = (df_full.isna().sum() / df_full.shape[0]) > feature_threshold
columns_name = columns_need_attention.index[columns_need_attention.values].to_list()
print("These features have over {}% missing values: ".format(feature_threshold * 100), end='')
print(*columns_name, sep=', ')
df_full = df_full.drop(columns=columns_name)

#%%
# Drop people with negative or zero times
insta_with_non_pos_time = df_full.time < 1
df_full = df_full[~insta_with_non_pos_time]
df_full.reset_index(drop=True, inplace=True)

# instance_threshold = 0.5
# rows_need_attention = (df_full.isna().sum(axis=1) / (df_full.shape[1] - 2)) > instance_threshold
# df_full = df_full[~rows_need_attention]
# df_full.reset_index(drop=True, inplace=True)

#%%
df_full.fillna(df_full.median(), inplace=True)

#%%
clin_inds = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2', 'glucose', 'hematocrit',
             'hemoglobin', 'platelets', 'wbc', 'aniongap', 'bicarbonate', 'bun', 'calcium', 'chloride', 'creatinine',
             'glucose_1', 'sodium', 'potassium', 'inr', 'pt', 'ptt']
for clin_ind in clin_inds:
    df_full[clin_ind + '_range'] = df_full[clin_ind + '_max'] - df_full[clin_ind + '_min']
    # print("Add {}, now the feature number is {}.".format(clin_ind, df_full.shape[1]))

df_full = df_full.reindex(sorted(df_full.columns), axis=1)

df_full.to_csv("data/MIMIC/MIMIC_IV_all_cause_failure.csv", index=False)