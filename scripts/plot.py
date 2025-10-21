import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, "same") / np.convolve(z, y, "same")


df = pd.read_csv("runs/ZoneEnv/tmp/logs.csv")
df["return"] = smooth(df["return"], 10)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(data=df, x="timestep", y="return")
plt.show()
