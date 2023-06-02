import pandas as pd
from Utils.evaluation_metrics import pitchCorr_f

file_name = "/scratch/ardas/Evaluation/Objective/objective_demo_alt_withPitch_epoch60_objective.csv"
df = pd.read_csv(file_name)

for pos in range(df.shape[0]):
    df.loc[pos, "Pitch Corr"] = pitchCorr_f(df.loc[pos, "source wav"], df.loc[pos, "converted wav"])

df.to_csv(file_name, index=False)
print("End")