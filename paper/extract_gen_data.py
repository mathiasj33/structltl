import pandas as pd

methods = ["struct_ltl", "genz_ltl", "deep_ltl", "ltl2action"]
depths = [2, 4, 8, 12]

data = []

for method in methods:
    for depth in depths:
        df = pd.read_csv(f"runs/WarehouseEnv/{method}/main/results_reach-{depth}.csv")
        agg = df.groupby("formula")["return"].agg(["mean"]).reset_index()
        overall_mean = agg["mean"].mean()
        data.append({"Method": method, "Depth": depth, "Mean Return": overall_mean})

df = pd.DataFrame(data)
print(df)
df.to_csv("extracted_means.csv", index=False)
