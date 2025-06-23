# Task 1: Data Cleaning & Preprocessing – AI & ML Internship

This repository contains my solution for Task 1 of the Elevate Labs AI/ML Internship program.

## 📌 Objective
Clean and preprocess the Titanic dataset to prepare it for machine learning.

---

## ✅ Steps Performed

1. **Loaded the Titanic dataset** from `train.csv`
2. **Explored** basic info, checked for missing values and data types
3. **Handled missing values**:
   - Filled missing `Age` with median
   - Filled missing `Embarked` with mode
   - Dropped `Cabin` (too many nulls)
4. **Encoded categorical variables**:
   - Used one-hot encoding for `Sex` and `Embarked`
5. **Standardized numerical columns** (`Age`, `Fare`) using `StandardScaler`
6. **Visualized outliers** in `Fare` using a boxplot
7. **Removed outliers** where `Fare` was greater than 3 (after scaling)
8. **Saved** the cleaned dataset as `cleaned_titanic.csv`

---

## 🛠 Tools Used
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

---

## 📁 Files in This Repo
- `titanic_preprocessing.py` – Full preprocessing code
- `train.csv` – Original dataset (optional)
- `cleaned_titanic.csv` – Final cleaned output
- `README.md` – This file

---

## 🎓 Learning Outcomes
- Learned data exploration and missing value imputation
- Understood encoding strategies and feature scaling
- Practiced visualizing and removing outliers
- Built a clean dataset ready for machine learning
