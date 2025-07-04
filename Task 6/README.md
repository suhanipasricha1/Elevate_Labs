# Task 6 - K-Nearest Neighbors (KNN) Classification

## 📑 Task Objective
The objective of this task is to understand and implement the **K-Nearest Neighbors (KNN)** algorithm for a classification problem using the Iris dataset.

## 🔧 Tools & Libraries Used
- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy

## 📂 Dataset
- **Iris Dataset**
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target: Iris Species (`Setosa`, `Versicolor`, `Virginica`)

## 🚀 Steps Performed
1. Loaded the dataset.
2. Normalized feature values using `StandardScaler`.
3. Split data into training and testing sets.
4. Implemented KNN classifier using `KNeighborsClassifier` from sklearn.
5. Experimented with different values of K.
6. Evaluated the model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
7. Visualized:
   - K vs Accuracy curve
   - Confusion Matrix heatmap
8. Saved screenshots of the results.

## 📈 Results
- Achieved **100% accuracy** on the test dataset.
- Perfect classification as shown in the confusion matrix and classification report.

## 🗂️ File Structure
```
Task6/
├── Iris.csv
├── KNN_Classifier.py
├── screenshots/
│   ├── k_vs_accuracy.png
│   └── confusion_matrix.png
└── README.md
```

## 🔍 How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn seaborn
   ```
2. Run the script:
   ```bash
   python KNN_Classifier.py
   ```
3. Check the output plots in the `screenshots/` folder.

## 📚 Interview Preparation
- How does KNN work?
- How to choose the right K?
- Why is normalization important in KNN?
- Pros and cons of KNN.
- Role of distance metrics (Euclidean, Manhattan, etc.).
- KNN’s time complexity and sensitivity to noise.

## 🏁 Conclusion
This task helped solidify the understanding of instance-based learning, distance metrics, model evaluation, and hyperparameter tuning using K.
