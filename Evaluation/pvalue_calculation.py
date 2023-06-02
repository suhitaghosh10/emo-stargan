import pandas as pd
from scipy import stats
base_path = "/scratch/ardas/Evaluation/Objective/objective_baseline_withPitch_new_objective_update.csv"
df_base = pd.read_csv(base_path)
df_base.reset_index()
ours_path = "/scratch/ardas/Evaluation/Objective/objective_Demo_alt_style_con_objective_update.csv"
df_ours = pd.read_csv(ours_path)
df_ours.reset_index()


res = stats.ttest_rel(df_base['speaker similarity score'].tolist(), df_ours['speaker similarity score'].tolist(), nan_policy="omit")
print(res)