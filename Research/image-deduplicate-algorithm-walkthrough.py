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
# # Image Deduplication Algorithm
# The purpose of this notebook is to go into detail as to how the image deduplication algorithm works. The content in this notebook is organized as follows:
#
# 1. Dataset Preparation
# 2. Loading The Model
# 3. Forwarding An Image Through The Model
# 4. Feature Extraction
# 5. Dimensionality Reduction - PCA
# 6. Clustering - DBSCAN
# 7. Inspecting Results

# +
import shutil 
import os

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict

from sklearn.decomposition import PCA
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import Model
from sklearn.cluster import DBSCAN
# -

# # 1. Dataset Preparation
# The dataset used in this notebook are images takens from Caltech101. This dataset contains images of objects belonging to 101 categories with about 50 images in each category. This dataset is pretty straightforward as most of the images do not have much background in them, only a close up shot of the object. The purpose of this dataset is to establish the proof of concept for the image deduplication algorithm.

# First we'll need to download the dataset. The entire dataset (15000+ images) can be downloaded [here](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz). After downloading the tar.gz file, put it in the folder alongside this python notebook. To unzip the tar.gz file, open a terminal window and change its directory to the folder and type in the command shown below.
#
# `tar -xf 101_ObjectCategories.tar.gz`
#
# Next, we'll need to convert the names of images downloaded. We can do so with the code in the cells below.

# + colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="JLJbcv0p0hv8" outputId="da863ef5-958e-47f8-c35e-c3f100a714a4"
folder_path = './101_ObjectCategories/'
all_folders = [name for name in os.listdir(folder_path) if name != '.DS_Store'] 

def edit_image_names(folder_path):
    '''
    Adds the name of the object infront of the image name
    Eg: image_0001.jpg -> barrel_image_0001.jpg
    '''
    for i in range(len(all_folders)):
        photos = [name for name in os.listdir(folder_path + all_folders[i])]
        for p in photos:
            os.rename(folder_path + all_folders[i] + '/' + p, 
                      folder_path + all_folders[i] + '/' + all_folders[i] + '_' + p)
            
edit_image_names(folder_path)
# -

# Next, you'll need to create a folder `ALL_PICTS` located beside this python notebook and 101_ObjectCategories folder. We'll be congregating all the images from the different folders into here.

# +
all_picts = './ALL_PICTS/'
def congregate_images(from_path, all_folders, to_path):
    '''
    Moves all images from separate folders into 1 main folder
    '''
    for i in range(len(all_folders)):
        photos = [name for name in os.listdir(folder_path + all_folders[i])]
        for p in photos:
            src_dir = from_path + all_folders[i] + '/'+ p
            dst_dir = to_path
            shutil.move(src_dir, dst_dir)

            
congregate_images(from_path = folder_path,
                  all_folders = all_folders,
                  to_path = all_picts)

# + [markdown] colab_type="text" id="u5y_fgURtJk4"
# # 2. Loading The Model
# -

# Now that our dataset has been prepared, we can move on to loading a pre-trained model from Tensorflow. Here we use the VGG16 model.

# + colab={"base_uri": "https://localhost:8080/", "height": 156} colab_type="code" id="XLZTcsEbtJlE" outputId="a09f293a-be4c-43f5-ab8f-9212d5bd7ec9"
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
model.summary()


# + [markdown] colab_type="text" id="AUfA0gvMtJlM"
# The summary gives us a layer-by-layer description of the network. Notice that VGG16 is a deep network with 13 convolutional layers. It was previously trained on millions of images, and has over 100,000,000 weights and biases, the majority of which connect to the first fully-connected layer (fc1). VGG-16 is setup to take a fixed-size (224 x 224 x 3) RGB image at its input, and then forward it through a series of altrnating convolutional and max-pooling layers, then capped off by three fully-connected layers of 4096, 4096, and 1000 neurons, where the last layer is our softmax classification layer.
#
# Notice that the output shape at each layer has `None` the first dimension. This is because the network can process multiple images in a single batch. So if you forward 5 images at shape [5, 224, 224, 3], then the output shape at each layer will be 5 in the first dimension.

# + [markdown] colab_type="text" id="E0KtPLZTtJlN"
# # 3. Forwarding An Image Through The Model
#
# In order to input an image into the network, it has to be pre-processed into a feature vector of the correct size. To help us do this, we will create a function `load_image(path)` which will handle the usual pre-processing steps: load an image from our file system and turn it into an input vector of the correct dimensions, those expected by VGG16, namely a color image of size 224x224.

# + colab={} colab_type="code" id="SqOr-Bv6tJlO"
def load_image(path):
    '''Takes in an image path
       Returns the data vector of the image and the image vector representation with specified dimensions (according to VGG16)'''
    
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# + [markdown] colab_type="text" id="78pMyAjitJlR"
# Lets use an example to demonstrate this. We'll load an image of a rhino and take a look at its data vector.

# + colab={"base_uri": "https://localhost:8080/", "height": 320} colab_type="code" id="6_qJxjr5tJlS" outputId="2846e58b-f11a-4c4e-e68c-da305380b32a"
rhino_img_path = all_picts + 'rhino_image_0047.jpg'
img, x = load_image(rhino_img_path)
print("shape of x: ", x.shape)
print("data type: ", x.dtype)
plt.imshow(img)
plt.show()

# + [markdown] colab_type="text" id="dcyLe4rktJlV"
# The shape of the image is [1, 224, 224, 3]. The reason it has the extra first dimension with 1 element is that the network can take batches of images to process them all simultaneously. So for example, 10 images can be propagated through the network if `x` has a shape of [10, 224, 224, 3], but this is not very important for our use case so just taking a mental note of it will do.
# -

# # 4. Feature Extraction

# + [markdown] colab_type="text" id="oqvb2ZsOtJlb"
# What we have in the `model` variable is a highly effective image classifier trained on the ImageNet database. We expect that the classifier must form a very effective representation of the image in order to be able to classify it with such high accuracy. We can use this to our advantage by re-purposing this for another task.
#
# What we do is we copy the model, but remove the last layer (the classification layer), so that the final layer of the new network, called `feat_extractor` is the second 4096-neuron fully-connected layer, "fc2 (Dense)".
#
# The way we do this is by instantiating a new model called `feature_extractor` which takes a reference to the desired input and output layers in our VGG16 model. Thus, `feature_extractor`'s output is the layer just before the classification, the last 4096-neuron fully connected layer.

# + colab={"base_uri": "https://localhost:8080/", "height": 901} colab_type="code" id="fcErc59xtJlc" outputId="4d477e2a-0c1a-438f-eab8-55f204831d39"
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

# + [markdown] colab_type="text" id="s0IXb0XUtJlg"
# Now let's see the feature extractor in action. We pass the same image from before into it, and look at the results. The `predict` function returns an array with one element per image (in our case, there is just one). Each element contains a 4096-element array, which is the activations of the last fully-connected layer `fc2` in VGG16.

# + colab={"base_uri": "https://localhost:8080/", "height": 282} colab_type="code" id="O4_JKCwstJlh" outputId="d0367895-466f-4071-f3d4-4207d0a377ef"
img, x = load_image(rhino_img_path)
feat = feat_extractor.predict(x)

plt.figure(figsize=(16,4))
plt.plot(feat[0])
plt.show()

# + [markdown] colab_type="text" id="wF_lKFKWtJll"
# Our expectation is that the `fc2` activations form a very good representation of the image, such that similar images should produce similar activations. In other words, the `fc2` activations of two images which have similar content should be very close to each other. We can exploit this to do information retrieval. 
# -

# Now that we know we can convert images to its feature representations using `feat_extractor`, we're almost ready to find similar looking images. But first, since the dataset we're dealing with here is pretty large in size, let's take a subset of this dataset by randomly selecting a bunch of images.
#
#
# In the next cell, we will open the folder `ALL_PICTS` and randomly select images till `max_num_images` has been achieved.

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="111VOoZltJln" outputId="e6492292-ef96-4e7e-ba20-c0a132edf425"
max_num_images = 1000

image_extensions = ['.jpg', '.png', '.jpeg', '.bmp']   # case-insensitive
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(all_picts) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]

if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]
    
print(f'Number of images randomly selected: {len(images)}')

# + [markdown] colab_type="text" id="tZa9ETq3tJlq"
# In the next cell, we will begin a loop which will open each image, extract its feature vector, and append it to a list called `features` which will contain our activations for each image. This process may take awhile depending on your setup so you may need to leave it running for a few minutes.

# +
features = []

for i, image_path in enumerate(images):
    if i%100 == 0:
        print(f'now at image {i}...') 
        
    img, x = load_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
# -

# # 5. Dimensionality Reduction - PCA

# + [markdown] colab_type="text" id="KUqhGryMtJlv"
# Alone, these activations provide a good representation, but it is a good idea to do one more step before using these as our feature vectors, which is to do a [principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce the dimensionality of our feature vectors. 
#
# #### Why PCA?
# Firstly, the 4096 dimension feature vector may have some redundancy in it, such that multiple elements in the vector are highly correlated or similar. This would skew similarity comparisons towards those over-represented features. 
#
# Secondly, operating over 4096 elements is inefficient both in terms of space/memory requirements and processor speed, and it would be better for us if we can reduce the length of these vectors but maintain the same effective representation. PCA allows us to do this by reducing the dimensionality down of the feature vectors from 4096 to much less, but maintain a representation which is still faithful to the original data, by preserving the relative inter-point distance.
#
# Thus, PCA reduces the amount of redundancy in our features (from duplicate or highly-correlated features), speeds up computation over them, and reduces the amount of memory they take up. 
#
# #### Explained Variance
# We know that each feature vector now has 4096 dimensions and we want to reduce this number by using PCA. But how do we know what is a good number of dimensions to choose? We can select the ideal number using the concept of explained variance.
#
# The PCA object allows users to select the number of components by either arbitarily selecting a number, OR by specifying a float that represents explained variance, which is the approach that we're interested in. If the argument entered into the PCA object is a float, the PCA object is intelligent enough to select the number of components that accounts for the % of variation as specified by explained variance.
#
# The next cell will instantiate a PCA object, which we will then fit our data to, choosing to keep the number of components that accounts for 95% of the variation in the data.

# +
explained_var = 0.95

features = np.array(features) # need to convert features to np array before inputting to PCA
pca = PCA(n_components = explained_var)
pca_features = pca.fit_transform(features)

print(f'Number of images that were converted to PCA features: {pca_features.shape[0]}')
print(f'Number of principal components selected: {pca_features.shape[1]}')
# -

# As you can see above, the number of components selected from specifying an explained variance of 0.95 is significantly lesser than the original 4096 dimensions. Again, just to reiterate, the number of components chosen (as seen in the printout) accounts for 0.95 of the variation in the data.

# # 6. Clustering - DBSCAN
# Now that we have fully prepared our data (converted images to feature vectors and finished dimensionality reduction), we're ready to find similar looking (near duplicate) images. The idea here is to find similar images by comparing their feature vectors, or more particularly, where these feature vectors exist in the dimensional space.
#
# We can do so using a clustering algorithm known as [Density-Based Spatial Clustering of Applications with Noise](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), or DBSCAN for short. Essentially what DBSCAN does is it groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points for a group to be considered a cluster. Outliers, points that do not belong to any clusters, are marked as -1 and points that fall into clusters will be marked as their respective arbitary cluster IDs.
#
# There are 2 key parameters of the DBSCAN algorithm:
# #### Epsilon (eps)
# - The maximum distance between two points for one to be considered as in the neighborhood of the other. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function
# - Increasing eps value makes it easier to form clusters
#
# #### Minimum Samples (min_samples)
# - The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. We will be using 2 here since we're interested in detecting even 2 similar looking images.

# #### Choosing Eps Value With PCA Features Hyperparameter
# Now, we know that we want to use the DBSCAN algorithm with argument for min_samples = 2. But how do we decide what eps should be since this is an arbitarily decided upon number? We can decide what an ideal eps value is by calculating it as a percentage of the hyperspace diameter of our PCA features.
#
# We first calculate the hyperspace parameter of our PCA features by adding up all the explained variance and taking the square root of that (Pythagoras theorem). Then take an arbitarily decided upon hyperspace_distance_percentage and multiply that by the hyperspace parameter.

# +
# Select hyperspace distance percentage
hyperspace_distance_percentage = 0.1
print(f"hyperspace_distance_percentage: {hyperspace_distance_percentage}")

# Calculate hyperspace diameter
total = 0
for i in pca.explained_variance_:
    total += i**2
hyperspace_diameter = math.sqrt(total)
print(f"hyperspace diameter: {hyperspace_diameter}")

# Calculate neighborhood distance based on percentage of hyperspace diameter
neighborhood_dist = hyperspace_distance_percentage * hyperspace_diameter
print(f"neighborhood_dist: {neighborhood_dist}")
# -

# Now we're ready to use the DBSCAN model.

dbscan_clustering = DBSCAN(eps = neighborhood_dist, min_samples = 2).fit(pca_features)
dbscan_clustering.labels_

# Recall that images that dont have neighbors are marked as -1 and images that do are marked as their respective cluster IDs. Although this dataset does not contain near duplicate images, we can expect that images that are labelled to belong to the same cluster should contain the same object and hence are considered as similar images.
#
# The code below converts the array of clustering labels to a dictionary and removes all images that do not belong to any cluster.

# +
# Convert clustering labels to a dictionary 
d = defaultdict(list)
for i, n in enumerate(dbscan_clustering.labels_):
    d[n].append(i)

del d[-1] # remove all images that dont belong to any clusters
d


# -

# # 7. Inspecting Results

# Now the algorithm is finally completed. We can use the function `display_neighbors()` to see if the algorithm has indeed clustered images containing similar objects together.

def display_neighbors(cluster):
    '''Takes in a list of images and prints the images side by side'''
    pics = []
    for i in cluster:
        img = image.load_img(images[i])
        img = img.resize((int(img.width*100/img.height), 100))
        pics.append(img)
    
    output = np.concatenate([np.asarray(t) for t in pics], axis = 1)
    plt.figure(figsize = (16,12))
    plt.imshow(output)
    plt.show()
    return


display_neighbors(d[12])

display_neighbors(d[3])

# --- END OF NOTEBOOK ---
