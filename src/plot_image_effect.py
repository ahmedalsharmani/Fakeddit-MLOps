import pandas as pd
import matplotlib.pyplot as plt

# Load probe results
df = pd.read_csv("probe_results.csv")

plt.figure(figsize=(8,5))
plt.hist(df["ok_delta"], bins=20, alpha=0.6, label="Correct Image Δ", color="green")
plt.hist(df["bad_delta"], bins=20, alpha=0.6, label="Mismatched Image Δ", color="red")
plt.axvline(df["ok_delta"].mean(), color="green", linestyle="--")
plt.axvline(df["bad_delta"].mean(), color="red", linestyle="--")
plt.title("Effect of Image on Model Confidence (Δ)")
plt.xlabel("Δ = |Confidence(with image) - Confidence(text only)|")
plt.ylabel("Number of Samples")
plt.legend()
plt.tight_layout()
plt.show()
