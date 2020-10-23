import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def compute_manhattan_distance(point, centroid):
    return np.sum(np.abs(point-centroid))

def cosine_distance(vector1, vector2):
    return 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2)))

def jarcard_distance(vector1, vector2):
    generalized_jarcard_similarity = np.sum(np.minimum(vector1, vector2))/np.sum(np.maximum(vector1, vector2))
    return 1 - generalized_jarcard_similarity

def calculate_SSE(center, members):
    length = len(members)
    sum=0
    for i in range(0,length):
        sum += np.sum((center-members[i])**2)

    return sum

########################################################################################################################
## TASK 01: ##
########################################################################################################################
# Initializing centroids:
centroids = np.array([[3,2],[4,8]])

total_iteration = 100
data_points = np.array([[3,5], [3,4], [2,8], [2,3], [6,2], [6,4], [7,3], [7,4], [8,5], [7,6]])

# Iterating K-means:
label = []
cluster_label = []
total_points = len(data_points)
k = len(centroids)

label = np.zeros(data_points.__len__())
for iteration in range(0, total_iteration):
    new_centroids = np.zeros(centroids.shape)
    centroid_counter = np.zeros(centroids.__len__())
    for index_point in range(0, total_points):
        distance = []
        for centroid_index in range(0, k):
            distance.append(compute_manhattan_distance(data_points[index_point], centroids[centroid_index]))
        label[index_point] = np.argmin(distance)
        new_centroids[int(label[index_point])] = new_centroids[int(label[index_point])] + data_points[index_point]
        centroid_counter[int(label[index_point])] += 1

    for i in range(0,k):
        new_centroids[i] = new_centroids[i]/centroid_counter[i]

    print("Pass:", iteration, "\n", new_centroids, "\n\n")

    # Breaking loop if center position does not change:
    if np.sum(np.abs(centroids-new_centroids)) == 0 :
        break
    centroids = new_centroids

# Scatter plot:
print("Centroids: \n", new_centroids)

i=0
cluster1 = []
cluster2 = []
for points in data_points:
    if label[i] == 0:
        cluster1.append(points)
    elif label[i] == 1:
        cluster2.append(points)
    else:
        print("Fetal Error: ")
    i+=1

cluster1 = np.vstack(cluster1)
cluster2 = np.vstack(cluster2)
plt.scatter(cluster1[:,0], cluster1[:,1], c='b')
plt.scatter(cluster2[:,0], cluster2[:,1], c='r')
plt.scatter(new_centroids[:,0], new_centroids[:,1], c='k')
plt.xlim(0,10)
plt.ylim(0,10)

########################################################################################################################
## TASK 02: ##
########################################################################################################################

# Importing and preprocessing Iris dataset:
data_df = pd.read_csv('iris.data')
X_data_df = data_df[["feature_1", "feature_2", "feature_3", "feature_4"]]
X_data = X_data_df.to_numpy()
le = LabelEncoder()
y_labels = le.fit_transform(data_df['labels']) # 0:Iris-setosa, 1:Iris-versicolor, 2:Irisvirginica

# Initializing centroids:
centroids = np.array([np.min(X_data,axis=0)+np.max(X_data,axis=0)/2,np.min(X_data,axis=0), np.max(X_data,axis=0)])

total_iteration = 100
data_points = X_data
# Iterating K-means:
SSE = 0
previous_SSE = 10000000
SSE_all = []
label = []
cluster_label = []
total_points = len(data_points)
k = len(centroids)
label = np.zeros(data_points.__len__())

start = time.time()
for iteration in range(0, total_iteration):
    new_centroids = np.zeros(centroids.shape)
    centroid_counter = np.zeros(centroids.__len__())
    for index_point in range(0, total_points):
        distance = []
        for centroid_index in range(0, k):
            distance.append(jarcard_distance(data_points[index_point], centroids[centroid_index]))
        label[index_point] = np.argmin(distance)
        new_centroids[int(label[index_point])] = new_centroids[int(label[index_point])] + data_points[index_point]
        centroid_counter[int(label[index_point])] += 1

    for i in range(0,k):
        new_centroids[i] = new_centroids[i]/centroid_counter[i]

    print("Pass:", iteration, "\n", new_centroids, "\n\n")

    # Breaking loop if center position does not change:
    if np.sum(np.abs(centroids-new_centroids)) == 0 :
        break

    # # Breaking the loop if SSE increases:
    # vote_cluster = np.zeros([3, 3])
    # i = 0
    # cluster1 = []
    # cluster2 = []
    # cluster3 = []
    #
    # for points in data_points:
    #     if label[i] == 0:
    #         vote_cluster[0, y_labels[i]] += 1
    #         cluster1.append(points)
    #     elif label[i] == 1:
    #         vote_cluster[1, y_labels[i]] += 1
    #         cluster2.append(points)
    #     elif label[i] == 2:
    #         vote_cluster[2, y_labels[i]] += 1
    #         cluster3.append(points)
    #     else:
    #         print("Fetal Error: ")
    #     i += 1
    #
    # cluster1 = np.vstack(cluster1)
    # cluster2 = np.vstack(cluster2)
    # cluster3 = np.vstack(cluster3)
    #
    # SSE = calculate_SSE(new_centroids[0], cluster1) + calculate_SSE(new_centroids[1], cluster2) + calculate_SSE(new_centroids[2], cluster3)
    # SSE_all.append(SSE)
    # if SSE >= previous_SSE:
    #     break

    previous_SSE = SSE
    centroids = new_centroids

end = time.time()

# Voting class:
vote_cluster = np.zeros([3,3])
i=0
cluster1 = []
cluster2 = []
cluster3 = []

for points in data_points:
    if label[i] == 0:
        vote_cluster[0,y_labels[i]] += 1
        cluster1.append(points)
    elif label[i] == 1:
        vote_cluster[1, y_labels[i]] += 1
        cluster2.append(points)
    elif label[i] == 2:
        vote_cluster[2, y_labels[i]] += 1
        cluster3.append(points)
    else:
        print("Fetal Error: ")
    i+=1

cluster1 = np.vstack(cluster1)
cluster2 = np.vstack(cluster2)
cluster3 = np.vstack(cluster3)

cluster1_ground_label = np.argmax(vote_cluster[0])
cluster2_ground_label = np.argmax(vote_cluster[1])
cluster3_ground_label = np.argmax(vote_cluster[2])

total_error = 0
x=0
for i in range(0,len(label)):
    if label[i]==0 and y_labels[i] != cluster1_ground_label:
        total_error +=1
    elif label[i] == 1 and y_labels[i] != cluster2_ground_label:
        total_error += 1
    elif label[i] == 2 and y_labels[i] != cluster3_ground_label:
        total_error += 1
    else:
        x+=1

# Results:
print("Centroids: \n", new_centroids)
print("Labels: \n", label)
print("Accuracy: ", 100- (total_error/len(label)*100))
total_SSE = calculate_SSE(new_centroids[0], cluster1) + calculate_SSE(new_centroids[1], cluster2) + calculate_SSE(new_centroids[2], cluster3)
print("SSE: \n",total_SSE)
print("Time Esapsed: ", end-start)
