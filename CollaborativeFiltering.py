# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 05:11:09 2018

@author: swathi
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

header = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('D:\RecommenderSystems\ml-100k\ml-100k\\u.data',sep = '\t',names = header)

print(df.head())

users_count = df.user_id.unique().shape[0]
movies_count = df.item_id.unique().shape[0]
print("no. of users: "+ str(users_count) + ",no. of movies: "+ str(movies_count))
train_data,test_data = cv.train_test_split(df,test_size=0.25)

user_item_matrix_train = np.zeros((users_count,movies_count))
for row in train_data.itertuples():
    user_item_matrix_train[row[1]-1,row[2]-1] = row[3]

user_item_matrix_test = np.zeros((users_count,movies_count))
for row in test_data.itertuples():
    user_item_matrix_test[row[1]-1,row[2]-1] = row[3]
   
user_sim = pairwise_distances(user_item_matrix_train,metric='cosine')
item_sim = pairwise_distances(user_item_matrix_train.T,metric='cosine') #transpose of above matrix

#predict normalizing the ratings
def predict_movie(ratings,sim,filter_type='u'):
    if(filter_type == 'u'):
        user_rating_mean = ratings.mean(axis=1)
        ratings_norm = (ratings - user_rating_mean[:,np.newaxis])
        prediction = user_rating_mean[:,np.newaxis] + sim.dot(ratings_norm)/np.array([np.abs(sim).sum(axis=1)]).T
    elif(filter_type == 'i'):
        prediction = ratings.dot(sim) / np.array([np.abs(sim).sum(axis=1)])
    return prediction

prediction_item = predict_movie(user_item_matrix_train, item_sim, filter_type='i')
prediction_user = predict_movie(user_item_matrix_train, user_sim, filter_type='u')

def calculate_mse(actual_rating,prediction):
    prediction = prediction[actual_rating.nonzero()].flatten()
    actual_rating = actual_rating[actual_rating.nonzero()].flatten()
    mse = sqrt(mean_squared_error(prediction, actual_rating))
    return mse

print ('RMSE user_based: ' + str(calculate_rmse(user_item_matrix_test,prediction_user)))
print ('RMSE item_based: ' + str(calculate_rmse(user_item_matrix_test,prediction_item)))
