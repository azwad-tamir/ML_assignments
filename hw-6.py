import numpy as np
from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import reader
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset:
ratings_df = pd.read_csv ('./ratings_small.csv')
rating_df_final = ratings_df[['userId','movieId','rating']]
# The ratings are in a scale of 1 to 5:
reader = reader.Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(rating_df_final, reader)


# Generating the different recommender system algorithms:
algo_PMF = SVD(biased=False) #PMF
algo_user_CF = KNNBasic(sim_options={'user_based':True}) #user based collaborative filtering
algo_item_CF = KNNBasic(sim_options={'user_based':False}) #item based collaborative filtering
algo_user_CF_MSD = KNNBasic(sim_options={'name':'MSD', 'user_based':True}) #user based CF with MSD similarity
algo_user_CF_cosine = KNNBasic(sim_options={'name':'cosine', 'user_based':True}) # user based CF with cosine similarity
algo_user_CF_pearson = KNNBasic(sim_options={'name':'pearson', 'user_based':True}) # user based CF with pearson similarity
algo_item_CF_MSD = KNNBasic(sim_options={'name':'MSD', 'user_based':False}) #item based CF with MSD similarity
algo_item_CF_cosine = KNNBasic(sim_options={'name':'cosine', 'user_based':False}) #item based CF with cosine similiarity
algo_item_CF_pearson = KNNBasic(sim_options={'name':'pearson', 'user_based':False}) #item based CF with pearson similarity


# Run 5-fold cross-validation and print results.
print("PMF model: ")
results_PMF = cross_validate(algo_PMF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("User based collaborative Filtering: ")
results_user_CF = cross_validate(algo_user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Item based Collaborative filtering: ")
results_item_CF = cross_validate(algo_item_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("User based collaborative Filtering with MSD similarity function:")
results_user_CF_MSD = cross_validate(algo_user_CF_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("User based collaborative Filtering with cosine similarity function:")
results_user_CF_cosine = cross_validate(algo_user_CF_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("User based collaborative Filtering with pearson similarity function:")
results_user_CF_pearson = cross_validate(algo_user_CF_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("Item based collaborative Filtering with MSD similarity function:")
results_item_CF_MSD = cross_validate(algo_item_CF_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Item based collaborative Filtering with cosine similarity function:")
results_item_CF_cosine = cross_validate(algo_item_CF_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Item based collaborative Filtering with pearson similarity function:")
results_item_CF_pearson = cross_validate(algo_item_CF_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Plotting results based on similarity function:
RMSE_UBCF = []
MAE_UBCF = []
RMSE_IBCF = []
MAE_IBCF = []

RMSE_UBCF.append(results_user_CF_MSD['test_rmse'].sum()/5)
RMSE_UBCF.append(results_user_CF_cosine['test_rmse'].sum()/5)
RMSE_UBCF.append(results_user_CF_pearson['test_rmse'].sum()/5)
MAE_UBCF.append(results_user_CF_MSD['test_mae'].sum()/5)
MAE_UBCF.append(results_user_CF_cosine['test_mae'].sum()/5)
MAE_UBCF.append(results_user_CF_pearson['test_mae'].sum()/5)
RMSE_IBCF.append(results_item_CF_MSD['test_rmse'].sum()/5)
RMSE_IBCF.append(results_item_CF_cosine['test_rmse'].sum()/5)
RMSE_IBCF.append(results_item_CF_pearson['test_rmse'].sum()/5)
MAE_IBCF.append(results_item_CF_MSD['test_mae'].sum()/5)
MAE_IBCF.append(results_item_CF_cosine['test_mae'].sum()/5)
MAE_IBCF.append(results_item_CF_pearson['test_mae'].sum()/5)
x= [1,2,3]
my_xticks = ['MSD','Cosine','Pearson']
plt.xticks(x, my_xticks)
plt.figure(1)
plt.plot(x, RMSE_UBCF, color='blue', marker='o', linestyle='solid', linewidth=1, markersize=6, label='RMSE_UBCF')
plt.plot(x, RMSE_IBCF, color='red', marker='o', linestyle='solid', linewidth=1, markersize=6, label='RMSE_IBCF')
plt.legend()
plt.show()

plt.figure(2)
plt.xticks(x, my_xticks)
plt.plot(x, MAE_UBCF, color='blue', marker='o', linestyle='solid', linewidth=1, markersize=6, label='MAE_UBCF')
plt.plot(x, MAE_IBCF, color='red', marker='o', linestyle='solid', linewidth=1, markersize=6, label='MAE_IBCF')
plt.legend()
plt.show()


results_user_CF_variable = []
results_item_CF_variable = []

# Exploring CF accuracies with different k values:
for k in range(1,202,2):
    algo_user_CF_variable = KNNBasic(k=k, sim_options={'user_based': True})
    print("User based Collarborative filtering with k=",k, ":")
    results_user_CF_variable.append(cross_validate(algo_user_CF_variable, data, measures=['RMSE', 'MAE'], cv=5, verbose=True))
    algo_item_CF_variable = KNNBasic(k=k, sim_options={'user_based': False})
    print("Item based Collarborative filtering with k=", k, ":")
    results_item_CF_variable.append(cross_validate(algo_item_CF_variable, data, measures=['RMSE', 'MAE'], cv=5, verbose=True))

# plotting results:
y_user_CF_rmse = []
y_item_CF_rmse = []
y_user_CF_mae = []
y_item_CF_mae = []
for i in range(0,len(results_user_CF_variable)):
    y_user_CF_rmse.append(results_user_CF_variable[i]['test_rmse'].sum()/5)
    y_item_CF_rmse.append(results_item_CF_variable[i]['test_rmse'].sum()/5)
    y_user_CF_mae.append(results_user_CF_variable[i]['test_mae'].sum()/5)
    y_item_CF_mae.append(results_item_CF_variable[i]['test_mae'].sum()/5)
X = range(1,202,2)
plt.figure(3)
plt.plot(X, y_user_CF_rmse, color='red', linestyle='solid', linewidth=1.5, label='RMSE_UBCF')
plt.plot(X, y_item_CF_rmse, color='blue', linestyle='solid', linewidth=1.5, label='RMSE_IBCF')
plt.legend()
plt.xlabel('k value')
plt.ylabel('RMSE')

plt.figure(4)
plt.plot(X, y_user_CF_mae, color='red', linestyle='solid', linewidth=1.5, label='RMSE_UBCF')
plt.plot(X, y_item_CF_mae, color='blue', linestyle='solid', linewidth=1.5, label='RMSE_IBCF')
plt.legend()
plt.xlabel('k value')
plt.ylabel('MAE')
plt.show()

print("Best k-value for UBCF: ", y_user_CF_rmse.index(min(y_user_CF_rmse)))
print("Best k-value for IBCF: ", y_item_CF_rmse.index(min(y_item_CF_rmse)))