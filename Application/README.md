# Application
This folder contains python scripts that run a local Flask server that contains the packaged up image deduplication API, /cluster_detector. Follow the steps below to set it up:

1. Set up virtual environment with a desired terminal and activate it
2. Install requirements.txt into your virtual environment
3. Type `python manage.py runserver` in the terminal with your virtual environment
4. Call for API /cluster_detector. The sample input and output are shown below.

<p align="center">
    <img width="1000" alt="api_cluster_detector_input_output" src="https://user-images.githubusercontent.com/56946413/128993211-358a64e0-c099-46b8-877c-02234e7b3725.png">
</p>

#### manage.py
This script runs the application. You will not need to make any edits here.

#### app.py
This script is the main application of this mini Flask server. There is only 1 API here which is /cluster_detector.

#### duplicate_image_detector.py
This is where the image deduplication algorithm is stored. When the /cluster_detector API is called, a helper function cluster_detector() residing here will be activated to identify similar looking images from the given images. This file only works with image URLs. However, if you intend to work with images from a local folder instead, replace this file with `duplicate_image_detector.py` from the `Misc` folder.

