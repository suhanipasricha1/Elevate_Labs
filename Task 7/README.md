# SVM Binary Classification Example

This project demonstrates how to:
- Load and prepare a dataset for binary classification.
- Train SVM models with linear and RBF kernels.
- Visualize decision boundaries using 2D data.
- Tune hyperparameters (`C` and `gamma`) using GridSearchCV.
- Evaluate performance with cross-validation.

## Dataset

We use the Breast Cancer Wisconsin dataset.

## Steps

1. **Preprocessing**: Dropping unnecessary columns and encoding labels.
2. **Training**: Two SVMs with different kernels.
3. **Visualization**: 2D decision boundary plots.
4. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation.

## Run

```bash
python svm_binary_classification.py
