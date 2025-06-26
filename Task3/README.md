# Linear Regression Task (Task 3)

## Objective
To implement and understand simple and multiple linear regression using Scikit-learn and evaluate the model using standard metrics.

## Dataset
We used the Boston Housing dataset, which contains features related to housing in Boston suburbs and the median price of owner-occupied homes.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Steps Performed
1. **Data Import & Preprocessing:** Loaded Boston Housing dataset, converted it into a Pandas DataFrame, and selected `RM` (average number of rooms) as the feature for simple linear regression.
2. **Train-Test Split:** Split the data into 80% training and 20% test set.
3. **Model Fitting:** Trained a `LinearRegression` model from `sklearn.linear_model`.
4. **Evaluation:** Calculated Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score to evaluate performance.
5. **Visualization:** Plotted actual vs predicted values along with the regression line.

## Results
The regression line clearly shows the relationship between number of rooms and house prices. The R² score indicates how well the model explains variance in the target variable.

---

For multiple regression, you can replace `X = df[['RM']]` with `X = df.drop('PRICE', axis=1)` to use all features.
