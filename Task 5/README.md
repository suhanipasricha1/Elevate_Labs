# Heart Disease Prediction Analysis

This project uses the **Heart Disease dataset** to train and evaluate tree-based models.

---

## 📂 Steps

1️⃣ **Load Data**  
- Reads `heart.csv` using pandas.

2️⃣ **Split Data**  
- Defines `X` (features) and `y` (`target` column).
- Splits into training and test sets (80% / 20%).

3️⃣ **Decision Tree Classifier**
- Trains a Decision Tree with default settings.
- Visualizes the tree up to depth 3 and saves it as `decision_tree.png`.

4️⃣ **Overfitting Analysis**
- Compares an unrestricted Decision Tree vs. a restricted one (`max_depth=4`).
- Prints training and test accuracy for both.

5️⃣ **Random Forest**
- Trains a Random Forest with 100 trees.
- Compares its accuracy to the Decision Tree.
- Prints training and test accuracy.

6️⃣ **Feature Importances**
- Displays feature importances from the Random Forest.

7️⃣ **Cross-Validation**
- Runs 5-fold cross-validation on the Random Forest.
- Prints average accuracy and standard deviation.

---

## 📊 Output

- Prints all results in the terminal.
- Saves the Decision Tree visualization as `decision_tree.png`.

---

## ▶️ Run

```bash
python heart_analysis.py
