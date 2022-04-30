# Import Libraries
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)
app.debug = True

# Import cluster detector function
from duplicate_image_detector import cluster_detector

# Default neighborhood distance
default_hyperspace_distance_percentage = 0.01 # Increase -> easier to find duplicates, recommended to stick between 0.02-0.09
default_explained_var = 0.98 # To select the number of principal components that explains X% of the variation -> typically btw 0.95-0.99

@app.route('/cluster_detector', methods = ['POST'])
def cluster_detector_call():
    '''
    Example of input json:
    {'hyperspace_distance_percentage': 0.01,
     'explained_var': 0.98,
     'images': 
        [
         {'id': '10381', 'image_path': 'https://cdn.getyourguide.com/5b03bf56bdc4f.jpeg/92.jpg'},
         {'id': '19384', 'image_path': 'https://media.tacdn.com/splice-spp-674x446/06/74/aa/fc.jpg'},
          ...]
        }
    '''
    try:
        content = request.json
        # Input default values for hyperspace_distance_percentage and explained_var if they were not given in input
        if 'hyperspace_distance_percentage' not in content : content['hyperspace_distance_percentage'] = default_hyperspace_distance_percentage
        if 'explained_var' not in content: content['explained_var'] = default_explained_var
        near_duplicates = cluster_detector(content['images'], content['hyperspace_distance_percentage'], content['explained_var'])
        return near_duplicates

    except Exception as e:
        output = (jsonify(code = 500, body = str(e)), 500)
        return output
