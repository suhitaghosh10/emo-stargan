import pandas as pd
from scipy import stats
#csv_path = "/scratch/ardas/Evaluation/Objective/objective_demo_alt_withPitch_epoch60_objective_update.csv"
csv_path = "/scratch/ardas/Evaluation/Objective/objective_baseline_withPitch_new_objective.csv"
df_esd = pd.read_csv(csv_path)
#df_esd = df_esd.loc[(df_esd["source dataset"]=="VCTK")]
#df_esd = df_esd.loc[(df_esd["emotion GT"]=="Neutral")]
#df_esd = df_esd.loc[(df_esd["source accent"]=="Canadian")]
#vctk_path = "/scratch/ardas/Evaluation/Objective/objective_ESD_demo_withPitch_objective.csv"
#df_vctk = pd.read_csv(vctk_path)
#df_esd = df_vctk.append(df_esd, ignore_index=True)
#df_esd = df_esd.loc[(df_esd["Model"]=="all")]
#df_esd = df_esd.loc[(df_esd["source gender"] == "Female") & (df_esd["target gender"] == "Female")]
#df_esd = df_esd_1
df_esd["emo_code_diff_100"] = df_esd["emo_code_diff"] * 100
df_esd["emo_alt_code_diff_100"] = df_esd["emo_alt_code_diff"] * 100
df_esd["PCC_100"] = df_esd["Pitch Corr"] * 100
#df_esd_all["PCC_100"] = df_esd_all["Pitch Corr"] * 100
"""for col in df_esd.columns:
    print(col)"""

#res = stats.ttest_rel(df_esd_all['PCC_100'], df_esd['PCC_100'])
#print(res)

acc_gt = ((df_esd["emotion GT"] == df_esd["converted emotion SVM"]).sum() / df_esd.shape[0]) *100
acc_svm = ((df_esd["source emotion SVM"] == df_esd["converted emotion SVM"]).sum() / df_esd.shape[0]) *100
print(acc_gt)
print(acc_svm)
print("Embed Mean", df_esd["emo_code_diff_100"].mean())
print("Embed Std", df_esd["emo_code_diff_100"].std())
print("Embed Alt Mean", df_esd["emo_alt_code_diff_100"].mean())
print("Embed Alt Std", df_esd["emo_alt_code_diff_100"].std())
print("PCC Mean", df_esd["PCC_100"].mean())
print("PCC Std", df_esd["PCC_100"].std())
print("PMOS Mean", df_esd["PMOS"].mean())
print("PMOS Std", df_esd["PMOS"].std())
print("CER Mean", df_esd["CER"].mean())
print("CER Std", df_esd["CER"].std())

