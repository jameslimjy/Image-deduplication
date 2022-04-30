from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import numpy as np
from collections import defaultdict
import math
import cv2
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove TF warnings

model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(
    inputs=model.input, outputs=model.get_layer("fc2").output)

def load_url_image(url):
    '''Takes in an image url and returns the resized image and image array'''
    img = io.imread(url)
    resized_img = cv2.resize(
        img, model.input_shape[1:3], interpolation=cv2.INTER_AREA)
    arr = image.img_to_array(resized_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return resized_img, arr

def cluster_detector(images, hyperspace_distance_percentage, explained_var):
    '''
    Takes in the following: 
        - images -> a list of dictionaries of images and ids
        - explained_var -> explained variance that instructs the PCA to select the number of components that accounts for explained_var in the data
        - hyperspace_distance_percentage -> the distance percentage of the hyperspace diameter that will be used as epsilon value in DBSCAN
    
    Outputs a dictionary of clusters
    
    Example of input:
    {'images': 
        [
         {'id': '10381', 'image_path': 'https://cdn.getyourguide.com/5b03bf56bdc4f.jpeg/92.jpg'},
         {'id': '19384', 'image_path': 'https://media.tacdn.com/splice-spp-674x446/06/74/aa/fc.jpg'},
          ...]
        }
     'hyperspace_distance_percentage': 0.01,
    'explained_var': 0.98,

    Example of output:
    {
    '0': [1, 5, 9],
    '1': [2, 31, 7],
    ...}
    
    '''

    features = []
    id_map = {}

    for idx, item in enumerate(images):
        # Convert dictionary of id : image_path to dictionary of list_id : {image_path: , id: }
        if idx%10 == 0:
            print(f'now at image {idx}...')

        id_map[idx] = {'image_path': item['image_path'], 'id': item['id']}
        # Convert images to feature vectors
        img, arr = load_url_image(item['image_path'])
        feat = feat_extractor.predict(arr)[0]
        features.append(feat)
        
    # Dimensionality reduction
    features = np.array(features)
    pca = PCA(n_components=explained_var)
    pca_features = pca.fit_transform(features)

    # Calculate hyperspace parameter
    total = 0
    for i in pca.explained_variance_:
        total += i**2
    hyperspace_diameter = math.sqrt(total)
    print(f"hyperspace diameter: {hyperspace_diameter}")
    print(f"hyperspace_distance_percentage: {hyperspace_distance_percentage}")
    
    # Calculate neighborhood distance based on percentage of hyperspace diameter
    neighborhood_dist = hyperspace_distance_percentage * hyperspace_diameter
    print(f"neighborhood_dist: {neighborhood_dist}")

    # Clustering
    cluster = DBSCAN(eps=neighborhood_dist, min_samples=2).fit(pca_features)
    d = defaultdict(list)
    for i, n in enumerate(cluster.labels_):
        d[int(n)].append(id_map[i]['id'])

    if -1 in d:
        del d[-1]
    return(dict(d))