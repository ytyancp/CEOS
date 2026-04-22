import pandas as pd
from scipy.stats import wilcoxon


df = pd.read_excel('../../results/vsLime_mlp.xlsx')
df_without_avg = df[:-1]    # remove avg row

lime_r2 = df_without_avg['LIME_R2']
ceos_r2 = df_without_avg['CEOS_R2']
lime_jaccard = df_without_avg['LIME_Jaccard']
ceos_jaccard = df_without_avg['CEOS_Jaccard']

statistic_r2, p_value_r2 = wilcoxon(lime_r2, ceos_r2, alternative='two-sided')
statistic_jaccard, p_value_jaccard = wilcoxon(lime_jaccard, ceos_jaccard, alternative='two-sided')

print(f"Wilcoxon signed-rank test for R²:")
print(f"LIME_R2 mean: {lime_r2.mean():.4f}")
print(f"CEOS_R2 mean: {ceos_r2.mean():.4f}")
print(f"P-value: {p_value_r2:.4f}")

print(f"Wilcoxon signed-rank test for Jaccard similarity:")
print(f"LIME_Jaccard mean: {lime_jaccard.mean():.4f}")
print(f"CEOS_Jaccard mean: {ceos_jaccard.mean():.4f}")
print(f"P-value: {p_value_jaccard:.4f}")
