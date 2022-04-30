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

# FOR LOCAL IMAGES
folder_path = '../../../../Pictures/tictag_beveragecans/' # edit accordingly
possible_extensions = ('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG')
images_paths = [i for i in os.listdir(folder_path) if i.endswith(possible_extensions)]
images_local = []
for i in range(len(images_paths)):
    images_local.append({"id": str(i), "image_path": images_paths[i]})

print(images_local)

def load_local_image(path):
    '''Takes in a local image and returns the resized image and image array'''
    url = folder_path+path
    img = cv2.imread(url) 
    resized_img = cv2.resize(
        img, model.input_shape[1:3], interpolation=cv2.INTER_AREA)
    arr = image.img_to_array(resized_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return resized_img, arr


def cluster_detector(images, hyperspace_distance_percentage, explained_var):
    features = []
    id_map = {}
    images = images_local 

    for idx, item in enumerate(images):
        # Convert dictionary of id:image_path to dictionary of list_id:image_path
        if idx%10 == 0:
            print(f'now at image {idx}...')

        id_map[idx] = item['image_path']
        # Convert images to feature vectors
        img, arr = load_local_image(item['image_path'])
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
        d[int(n)].append(id_map[i])

    if -1 in d:
        del d[-1]
    return(dict(d))
