# Industrial-Copper-Modeling

## **Process**

**1) Data Understanding:** 
- Identify the types of variables (continuous, categorical)
and their distributions. Some rubbish values are present in ‘Material_Reference’
which starts with ‘00000’ value which should be converted into null. Treat
reference columns as categorical variables. INDEX may not be useful.

**2) Data Preprocessing:**
  -  Handle missing values with mean/median/mode.
  - Treat Outliers using IQR or Isolation Forest from sklearn library.
  - Identify Skewness in the dataset and treat skewness with appropriate
data transformations, such as log transformation(which is best suited to
transform target variable-train, predict and then reverse transform it back
to original scale eg:dollars), boxcox transformation, or other techniques,
to handle high skewness in continuous variables.
  - Encode categorical variables using suitable techniques, such as one-hot
encoding, label encoding, or ordinal encoding, based on their nature and
relationship with the target variable.

**3) EDA:**
 
 - Try visualizing outliers and skewness(before and after treating skewness)
using Seaborn’s boxplot, distplot, violinplot.

**4) Feature Engineering:**
  - Engineer new features if applicable, such as aggregating
or transforming existing features to create more informative representations of
the data. And drop highly correlated columns using SNS HEATMAP.

**5) Model Building and Evaluation:**

- Split the dataset into training and testing/validation sets.

- Train and evaluate different classification models, such as
ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using
appropriate evaluation metrics such as accuracy, precision, recall, F1
score, and AUC curve.

- Optimize model hyperparameters using techniques such as
cross-validation and grid search to find the best-performing model.

- Interpret the model results and assess its performance based on the
defined problem statement.

- Same steps for Regression modelling.(note: dataset contains more noise
and linearity between independent variables so itll perform well only with
tree based models)

**6) Model GUI:** 

- Using streamlit module, create interactive page with
    - (1) task input( Regression or Classification) and
    - (2) create an input field where you can enter each column value except
‘Selling_Price’ for regression model and except ‘Status’ for classification
model.

    - (3) perform the same feature engineering, scaling factors, log/any
transformation steps which you used for training ml model and predict this new
data from streamlit and display the output.

**7) Tips:** 

- Use pickle module to dump and load models such as encoder(onehot/
label/ str.cat.codes /etc), scaling models(standard scaler), ML models. First fit
and then transform in separate line and use transform only for unseen data
