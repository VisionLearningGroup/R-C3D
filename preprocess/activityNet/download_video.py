# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import json
import os

annotation_file = open('activity_net.v1-3.min.json')
annotation = json.load(annotation_file)

video_database = annotation['database']
videos = annotation['database'].keys()

# Download the ActivityNet videos into the ./videos folder
command1 = 'mkdir '+'videos'
os.system(command1)

for i in videos:
    url = video_database[i]['url']
    command3 = 'youtube-dl -o '+'videos/'+i+' '+url
    print command3
    os.system(command3)



