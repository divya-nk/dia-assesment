#! /usr/bin/env python

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import f1_score



def calc_loan(df_qty, df_cost):
    df_total_cost = df_qty*df_cost
    return round(sum(df_total_cost),2)

if __name__ == '__main__':

    orig_purchase = pd.read_csv(sys.argv[1])
    next_purchase = pd.read_csv(sys.argv[2])
    cust_features = pd.read_csv(sys.argv[3])
    prod_features = pd.read_csv(sys.argv[4])
    last_month = pd.read_csv(sys.argv[5])
    next_month = pd.read_csv(sys.argv[6])
    
    #pre-processing favorite_genres:
    cust_features['favorite_genres'] = cust_features.favorite_genres.apply(lambda x: x.replace("-", ""))
    cust_features['favorite_genres'] = cust_features.favorite_genres.apply(lambda x: x.lower())
    
    #one-hot encoding favorite_genres
    vect = CountVectorizer()
    X = vect.fit_transform(cust_features.favorite_genres)
    cust_features = cust_features.join(pd.DataFrame(X.toarray(), columns=vect.get_feature_names()))
    
    #calculating last month's loan
    last_month_loan = calc_loan(orig_purchase['quantity_purchased'], orig_purchase['cost_to_buy'])
    #print('Loan taken for last month\'s purchase: ', last_month_loan)

    #calculating next month's cost
    next_month_cost = calc_loan(next_purchase['quantity_purchased'], next_purchase['cost_to_buy'])
    #print('Loan taken for next month\'s purchase: ', next_month_cost)
    
    #Calculate shipping cost for last month's assortment:
    last_month['shipping_cost'] = last_month.purchased.apply(lambda x: 0.6 if x==True else 1.2)
    
    last_month_shipping_cost = round(sum(last_month['shipping_cost']),2)
    #print('Last month assortment\'s shipping cost: ', last_month_shipping_cost)
    
    #Calculate total sales last month:
    df_last_month_sales = pd.merge(orig_purchase, last_month, on ='product_id')
    total_sales_last_month = round(sum(df_last_month_sales['retail_value'].where(df_last_month_sales['purchased']==True, 0)),2)
    
    # let's build models
    #preprocessing data for models

    data_last = pd.merge(df_last_month_sales, prod_features, on = 'product_id')
    data_last = pd.merge(data_last, cust_features, on = 'customer_id')
    data_last = data_last.drop(columns=['favorite_genres'])
    #data_last.columns
    
    #preprocessing and label encoding columns - preparing data for model:
    data_last['age_bucket'] = data_last['age_bucket'].astype(str)
    le = LabelEncoder()
    data_last['purchased'] = le.fit_transform(data_last['purchased'])
    data_last['fiction'] = le.fit_transform(data_last['fiction'])
    data_last['genre'] = le.fit_transform(data_last['genre'])
    data_last['age_bucket'] = le.fit_transform(data_last['age_bucket'])
    data_last['is_returning_customer'] = le.fit_transform(data_last['is_returning_customer'])
    
    
    predictors = ['retail_value', 'length', 'difficulty','fiction', 'genre', 'age_bucket', 
                'is_returning_customer', 'beachread', 'biography', 'classic', 'drama', 'history', 
                'poppsychology', 'popsci', 'romance', 'scifi', 'selfhelp', 'thriller']
    y = data_last['purchased']
    X = data_last[predictors]
    
    #Train, test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    #Ftting AdaBoostClassifier()
    abc = AdaBoostClassifier()
    #abc.fit(X_train, y_train)
    abc.fit(X, y)
   

    #calculate left-over from last_month
    df_temp = last_month.groupby(['product_id'])['purchased'].sum().reset_index()
    df_temp = pd.DataFrame(data = df_temp)
    
    orig_leftover = pd.merge(orig_purchase, df_temp, on = 'product_id')
    orig_leftover['qty_left']=orig_leftover['quantity_purchased'] - orig_leftover['purchased']
    orig_leftover = orig_leftover.drop(columns =['quantity_purchased', 'purchased'])

    #preparing data for prediction of next month's sales
    data_next = pd.merge(next_month, orig_leftover, on = 'product_id')
    data_next = pd.merge(data_next, cust_features, on = 'customer_id')
    data_next = pd.merge(data_next, prod_features, on = 'product_id')
    
    data_next['age_bucket'] = data_next['age_bucket'].astype(str)
    data_next['fiction'] = le.fit_transform(data_next['fiction'])
    data_next['genre'] = le.fit_transform(data_next['genre'])
    data_next['age_bucket'] = le.fit_transform(data_next['age_bucket'])
    data_next['is_returning_customer'] = le.fit_transform(data_next['is_returning_customer'])
    
    features_for_prediction = ['retail_value', 'length', 'difficulty', 'fiction', 'genre', 'age_bucket', 'is_returning_customer', 'beachread', 'biography',
       'classic', 'drama', 'history', 'poppsychology', 'popsci', 'romance',
       'scifi', 'selfhelp', 'thriller']
    X = data_next[features_for_prediction]
    
    #prediction of sales
    pred_purchase = abc.predict(X)
    #print('no of books predicted to be purchased: ',sum(pred_purchase))
    #sum(pred_purchase)/X.shape[0]
    
    #lets calculate shipping cost for next_month's prediction
    next_month_shipping = (sum(pred_purchase)*0.6 + (X.shape[0]-sum(pred_purchase)*1.2))
    #print('Shipping cost predictions for next month\'s assortment: ', next_month_shipping)
    
    #calculate sales for next month:
    data_next['pred_purchase'] = pred_purchase
    next_sales = round(sum(data_next['retail_value'].where(data_next['pred_purchase']==1, 0)),2)
    #print("Sale prediction for next month: ", next_sales)
    
    #coming to the actual question now! if we'll be able to pay back loan and afford next book purchase?
    total_cost_to_us = last_month_loan + next_month_cost + last_month_shipping_cost + next_month_shipping
    total_sales = total_sales_last_month + next_sales
    #print("Did we make it? How much: ", total_sales - total_cost_to_us)
    print('Yes' if (total_sales - total_cost_to_us>0) else 'No')
    
    
    
