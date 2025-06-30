import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv("heart.csv")

# 2. Define features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Decision Tree Classifier (unrestricted)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Visualize the tree up to depth 3 for clarity
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True, max_depth=3)
plt.title("Decision Tree (Visualized up to depth=3)")
plt.savefig("decision_tree.png")
plt.close()

# 5. Analyze overfitting: unrestricted vs. restricted
dt_unres = DecisionTreeClassifier(random_state=42)
dt_unres.fit(X_train, y_train)
train_acc_unres = accuracy_score(y_train, dt_unres.predict(X_train))
test_acc_unres = accuracy_score(y_test, dt_unres.predict(X_test))

dt_res = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_res.fit(X_train, y_train)
train_acc_res = accuracy_score(y_train, dt_res.predict(X_train))
test_acc_res = accuracy_score(y_test, dt_res.predict(X_test))

print(f"Unrestricted Decision Tree -> Train: {train_acc_unres:.2f}, Test: {test_acc_unres:.2f}")
print(f"Restricted Decision Tree (depth=4) -> Train: {train_acc_res:.2f}, Test: {test_acc_res:.2f}")

# 6. Train Random Forest and compare accuracy
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Random Forest -> Train: {rf_train_acc:.2f}, Test: {rf_test_acc:.2f}")

# 7. Feature importances (Random Forest)
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", feature_importance)

# 8. Cross-validation score for Random Forest
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\nRandom Forest Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
