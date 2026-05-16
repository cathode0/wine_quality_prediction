import matplotlib.pyplot as plt
import seaborn as sns

def correlation_plot(df):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def histograms(df):
    df.hist(figsize=(12, 10), bins=20)
    plt.tight_layout()
    plt.show()


def boxplots(df, features):
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='quality', y=feature, data=df)
        plt.title(f"{feature} vs Quality")
        plt.show()