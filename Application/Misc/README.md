# Misc

### duplicate_image_detector_local.py
This python script is similar in function to duplicate_image_detector.py, but it is used when working with images on a local folder instead. The API is called the same way with the same arguments, but the `folder_path` has to be edited accordingly. 

The API will still take in `hyperspace_distance_percentage` and `explained_var` as specified by the user, but `images` fed into the API are redundant since the images will be read via the local folder path instead.

Use this script if you are interested in testing what different outputs will arise from different arguments for a given set of images.
