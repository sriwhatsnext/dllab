

# Task 1: Dataset Exploration


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("/content/data_banknote_authentication.txt",header=None,names=[
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
])


print("First Five Samples")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDescriptive Statistics")
print(df.describe())

features = ["Variance", "Skewness", "Curtosis", "Entropy"]

for feature in features:

    plt.figure(figsize=(8,6))

    plt.hist(
        df[feature],
        bins=30,
        edgecolor='black'
    )

    plt.xlabel(feature, fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title(f"Histogram of {feature}", fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()

    plt.savefig(
        f"Histogram_{feature}.eps",
        format="eps",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()







plt.figure(figsize=(10,6))

sns.boxplot(data=df)

plt.xlabel("Features", fontsize=15)
plt.ylabel("Values", fontsize=15)
plt.title("Boxplot of Features", fontsize=15)

plt.xticks(rotation=30, fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()

plt.savefig(
    "Boxplot.eps",
    format="eps",
    dpi=600,
    bbox_inches="tight"
)

plt.show()






# CORRELATION HEATMAP

plt.figure(figsize=(8,7))

ax = sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size":15}
)

plt.title("Correlation Heatmap", fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()

# Save as PDF (NOT EPS)
plt.savefig(
    "Correlation_Heatmap.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()



# SCATTER PLOT FUNCTION


def save_scatter(x_feature, y_feature, filename):

    plt.figure(figsize=(8,6))

    class0 = df[df["Class"] == 0]
    class1 = df[df["Class"] == 1]

    plt.plot(
        class0[x_feature],
        class0[y_feature],
        '.',
        color='blue',
        markersize=2,
        label='Class 0'
    )

    plt.plot(
        class1[x_feature],
        class1[y_feature],
        '.',
        color='red',
        markersize=2,
        label='Class 1'
    )

    plt.xlabel(x_feature, fontsize=15)
    plt.ylabel(y_feature, fontsize=15)
    plt.title(f"{x_feature} vs {y_feature}", fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=12)

    plt.grid(True)

    plt.tight_layout()

    # Save as PDF
    plt.savefig(
        filename,
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()


# SCATTER PLOTS


save_scatter(
    "Variance",
    "Skewness",
    "Variance_vs_Skewness.pdf"
)

save_scatter(
    "Variance",
    "Curtosis",
    "Variance_vs_Curtosis.pdf"
)

save_scatter(
    "Variance",
    "Entropy",
    "Variance_vs_Entropy.pdf"
)

save_scatter(
    "Skewness",
    "Curtosis",
    "Skewness_vs_Curtosis.pdf"
)

save_scatter(
    "Skewness",
    "Entropy",
    "Skewness_vs_Entropy.pdf"
)

save_scatter(
    "Curtosis",
    "Entropy",
    "Curtosis_vs_Entropy.pdf"
)

import glob
import subprocess
import os

pdf_files = glob.glob("*.pdf")

for pdf in pdf_files:
    eps = os.path.splitext(pdf)[0] + ".eps"

    subprocess.run([
        "pdftops",
        "-eps",
        pdf,
        eps
    ])

print("Finished converting all PDFs to EPS.")



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assume last column is target

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Normalize

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert labels to binary (if necessary)

unique = np.unique(y)

if len(unique) == 2:
    y = np.where(y == unique[0], 0, 1)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs

    # Weight Initialization
    def initialize(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    # Step Activation Function
    def activation(self, x):
        return 1 if x >= 0 else 0

    # Forward Propagation
    def predict_sample(self, x):
        linear = np.dot(x, self.weights) + self.bias
        return self.activation(linear)

    # Training using Perceptron Learning Rule
    def fit(self, X, y):

        self.initialize(X.shape[1])

        # Initialize history only once
        self.errors = []
        self.weight_history = []
        self.bias_history = []

        for epoch in range(self.epochs):

            errors = 0

            for i in range(len(X)):

                prediction = self.predict_sample(X[i])

                update = self.lr * (y[i] - prediction)

                self.weights += update * X[i]
                self.bias += update

                if update != 0:
                    errors += 1

            # Store history after each epoch
            self.errors.append(errors)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            print("--------------------------------")
            print("Epoch:", epoch + 1)
            print("Misclassified Samples:", errors)
            print("Weights:", self.weights)
            print("Bias:", self.bias)

    def predict(self, X):
        return np.array([self.predict_sample(x) for x in X])

model = Perceptron(
    learning_rate=0.1,
    epochs=20
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nPrecision")
print(precision_score(y_test, y_pred))

print("\nRecall")
print(recall_score(y_test, y_pred))

print("\nF1 Score")
print(f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Training Error vs Epoch
# ==========================

plt.figure(figsize=(8,6))

plt.plot(
    range(1, len(model.errors)+1),
    model.errors,
    marker='o',
    linewidth=2
)

plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Misclassified Samples", fontsize=15)
plt.title("Training Error vs Epoch", fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.grid(True)

plt.tight_layout()

plt.savefig("Training_Error_vs_Epoch.eps", format="eps", dpi=600)

plt.show()


# ==========================
# Weight Evolution
# ==========================

weights = np.array(model.weight_history)

plt.figure(figsize=(8,6))

for i in range(weights.shape[1]):
    plt.plot(
        range(1, len(weights)+1),
        weights[:, i],
        marker='o',
        linewidth=2,
        label=f"Weight {i+1}"
    )

plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Weight Value", fontsize=15)
plt.title("Weight Evolution", fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=13)

plt.grid(True)

plt.tight_layout()

plt.savefig("Weight_Evolution.eps", format="eps", dpi=600)

plt.show()


# ==========================
# Bias Evolution
# ==========================

plt.figure(figsize=(8,6))

plt.plot(
    range(1, len(model.bias_history)+1),
    model.bias_history,
    marker='o',
    linewidth=2
)

plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Bias", fontsize=15)
plt.title("Bias Evolution", fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.grid(True)

plt.tight_layout()

plt.savefig("Bias_Evolution.eps", format="eps", dpi=600)

plt.show()


# ==========================
# Confusion Matrix
# ==========================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8,6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

ax.set_xlabel("Predicted Label", fontsize=15)
ax.set_ylabel("True Label", fontsize=15)
ax.set_title("Confusion Matrix", fontsize=15)

ax.tick_params(axis='both', labelsize=15)

# Set font size of confusion matrix values
for text in ax.texts:
    text.set_fontsize(15)

plt.tight_layout()

plt.savefig("Confusion_Matrix.eps", format="es", dpi=600)

plt.show()

# ==========================
# Learning Rate Comparison
# ==========================

learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(8,6))

for lr in learning_rates:

    model = Perceptron(
        learning_rate=lr,
        epochs=20
    )

    model.fit(X_train, y_train)

    plt.plot(
        range(1, len(model.errors)+1),
        model.errors,
        marker='o',
        linewidth=2,
        label=f"Learning Rate = {lr}"
    )

plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Misclassified Samples", fontsize=15)
plt.title("Learning Rate Comparison", fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=13)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Learning_Rate_Comparison.eps",
    format="eps",
    dpi=600
)

plt.show()

























