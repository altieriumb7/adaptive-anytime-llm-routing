import pandas as pd

df = pd.read_csv("results_abl/summary.csv")
df["RegressionRate"] = df["SolvedPct"] - df["AccStrict@4"]
df = df.sort_values("RegressionRate", ascending=False)

print(df[["MODEL","SolvedPct","AccStrict@4","RegressionRate","AUC_Strict"]].to_string(index=False))
