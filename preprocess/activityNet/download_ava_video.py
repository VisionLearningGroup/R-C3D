# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import json
import os

annotation_file = open('Activity.json')


for line in annotation_file.readlines():
    down = json.loads(line)
    url = down['url']
    filename = url.split('/')[-1]
    host = url.split('/')[0]
    command3 = 'curl http://xsio.qiniu.io/' + filename + ' -H \'Host:' + host + '\' -o videos/' + filename
    print command3
    os.system(command3)



