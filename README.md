#### Author: Muthudivya NK
#### Dia&Co technical assessment

## How to run this script 

python can_we_afford.py original_purchase_order.csv next_purchase_order.csv customer_features.csv product_features.csv last_month_assortment.csv next_month_assortment.csv


## Which machine learning model you chose to use and why?

As this is basically a classification problem, whether the customer will purchase the book or not. And depending on the prediction of the sales, further calculations of whether we can afford to pay off the loan taken for the last month and be able to afford to buy books for the next month is calculated.

Hence, I initially chose to fit few models and evaluate their performance on the given dataseet. Logistic Regression, AdaBoost Classifier and RandomForestClassifier being the classifiers I chose to run my initial tests on. Based on their performance I chose to go ahead to AdaBoost Classifier as it had the highest accuracy score of ~75% in comparison to others (LR: ~64%, RFC: ~72%)

75% is a decent accuracy score for this problem. Hence I proceeded with AdaBoostClassifier model for predicting sales for next month assortment.

## How you validated your model and why you chose to validate it in this way.

Since I already had a set of data of customers who have purchased books or not from the original purchase order. I used this dataset to fit my model and validate it. I split the dataset in 80:20 ratio and trained my model on the first split and validated it on the latter. This way, I was able to guage how the model is performing - i.e if it's able to make correct predictions on customer's option of buying the book or returning it.

### Contents:
README.md - this file
can_we_afford.py - File containing python script
scratchpad_divya.ipynb - Python Notebook containing work along with thought-process. Its not well detailed or well documented.

Datasets:
next_month_assortment.csv    product_features.csv
customer_features.csv      next_purchase_order.csv
last_month_assortment.csv  original_purchase_order.csv

