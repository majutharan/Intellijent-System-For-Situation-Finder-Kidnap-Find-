import jason as jason
import numpy as np
import cv2
import os

from pandas import json
from sightengine.client import SightengineClient




def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


full_file_paths = get_filepaths("/home/majutharan/Documents/projects/Final_research/KidnapFind/kidnap/Image_by_frame")

for f in full_file_paths:
    if f.endswith(".jpg"):
        myimage = f
        img = cv2.imread(myimage)
        client = SightengineClient('1298430622', '2e8zF7ha5hzz7SUVjmHZ')
        output = client.check('wad').set_file(myimage)
        print(output['weapon'])


