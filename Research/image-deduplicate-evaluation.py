# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="jw96lUzwtJk1"
# # Image Deduplicate Evaluation
# The purpose of this notebook is to go through various clustering evaluation metrics that can be used to evaluate the results of the image-deduplication algorithm. Since the algorithm utilizes clustering as its last step to identify similar images, we'll be looking at clustering evaluation metrics. 
#
# Evaluating the performance of a clustering algorithm is not as trivial as counting the number of errors or the precision and recall of a supervised classification algorithm. In particular any evaluation metric should not take the absolute values of the cluster labels into account but rather if this clustering define separations of the data similar to some ground truth set of classes or satisfying some assumption such that members belong to the same class are more similar than members of different classes according to some similarity metric.
#
# All clustering performance evaluation metrics can be seperated into 2 main camps: requires ground truth label assignments vs does not require ground truth label assignments. In production, you will rarely use the former since ground truth label assignments will hardly be available. However, when developing, you might opt to label a dataset via visual inspection so that you can use a more reliable evaluation metric that requires ground truth labels.
#
# Although there are many difference clustering evaluation metrics available, only a few are listed below. Refer to [here](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) for a more comprehensive list.
# -

from sklearn import metrics

# Get list of true and pred labels
df = # edit accordingly
labels_true = [] # edit accordingly
labels_pred = [] # edit accordingly

# # Ground Truth Labels Required

# ## Fowlkes-Mallows index

# The Fowlkes-Mallows index can be used when the ground truth class assignments of the samples is known. The Fowlkes-Mallows score FMI is defined as the geometric mean of the pairwise precision and recall.
#
# **FMI = TP / sqrt((TP + FP) * (TP + FN))**
#
# - TP is the number of True Positive (i.e. the number of pair of points that belongs in the same clusters in both labels_true and labels_pred)
# - FP is the number of False Positive (i.e. the number of pair of points that belongs in the same clusters in labels_true and not in labels_pred)
# - FN is the number of False Negative (i.e the number of pair of points that belongs in the same clusters in labels_pred and not in labels_True).
#
# The score ranges from 0 to 1. A high value indicates a good similarity between two clusters.
#
# #### Advantages
# - Random (uniform) label assignments have a FMI score close to 0.0 for any value of n_clusters and n_samples 
#
# - Upper-bounded at 1: Values close to zero indicate two label assignments that are largely independent, while values close to one indicate significant agreement. Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that the two label assignments are equal
#
# - No assumption is made on the cluster structure: can be used to compare clustering algorithms such as k-means which assumes isotropic blob shapes with results of spectral clustering algorithms which can find cluster with “folded” shapes
#
#
# #### Disadvantages
# - Requires ground truth labels, which will almost never be available in deployment

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="HQ8IRjEAtJk5" outputId="6e7dfd63-1919-4c2f-e7d1-ac0e70fd909b"
# Fowlkes-Mallows score
metrics.fowlkes_mallows_score(labels_true, labels_pred)
# -

# # Ground Truth Labels Not Required

# ## Silhouette Coefficient 

# The Silhouette Coefficient is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The Silhouette Coefficient ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. Ground truth not required.
#
# The Silhouette Coefficient is defined for each sample and is composed of two scores. The Silhouette Coefficient S for a single sample is then given as:
#
# **S = (B-A) / max(A,B)**
#
# - A: The mean distance between a sample and all other points in the same class.
# - B: The mean distance between a sample and all other points in the next nearest cluster.
#
#
# #### Advantages
# - The Silhouette Coefficient is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters
# - The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster
#
# #### Disadvantages
# - The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN
#

metrics.silhouette_score(df, labels_pred, metric='euclidean')
