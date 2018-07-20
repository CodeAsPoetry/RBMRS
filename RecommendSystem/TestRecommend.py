# coding="utf-8"

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__=="__main__":
    header = ['user_id','item_id','rating','timestamp']
    df = pd.read_csv('../ml-100k/u.data',sep='\t',names=header)
    # print(df)
    n_users = df.user_id.unique().shape[0]    # df.user_id.unique():<class 'numpy.ndarray'>
    n_items = df.item_id.unique().shape[0]
    # print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
    train_data, test_data = cv.train_test_split(df, test_size=0.25)

    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        # print(line)    # Pandas(Index=42465, user_id=299, item_id=81, rating=4, timestamp=889504036)
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    # print(train_data_matrix.shape)    # (943, 1682)
    # print(test_data_matrix.shape)     # (943, 1682)

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

    # print(user_similarity.shape)       # (943, 943)
    # print(item_similarity.shape)       # (1682, 1682)

    def predict(ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            # You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
                [np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred


    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')


    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))


    # print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    # print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    print('The sparsity level of MovieLens100K is ' + str(sparsity * 100) + '%')
