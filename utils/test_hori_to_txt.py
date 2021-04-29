import os
import json
import numpy as np

json_file = "rotated_teeth_test_coco_format.json" # => coco format
dir_name = 'input/horizontal-ground-truth'

with open(json_file) as f:
    detectron_json = json.load(f)

try:
    if not (os.path.isdir(dir_name)):
        os.makedirs(os.path.join(dir_name))
except OSError as e:
    if e.errno != e.errno.EXIST:
        print("Failed to create directory.")
        raise

for z in range(len(detectron_json['categories'])):
    detectron_json['categories'][z]['name'] = detectron_json['categories'][z]['name'].replace(" ", "_")

for i in range(len(detectron_json['images'])):
    txt_name = str(detectron_json['images'][i]['id']) + ".txt"
    txt = open( dir_name + '/' +txt_name, 'w')
    for j in range(len(detectron_json['annotations'])):
        text = ""
        if detectron_json['images'][i]['id'] == detectron_json['annotations'][j]['image_id']:
            text += str(detectron_json['categories'][detectron_json['annotations'][j]['category_id']]['name']) + " "
            text += str(detectron_json['annotations'][j]['bbox']) + " 0.0"
            text = text.replace('[', '')
            text = text.replace(']', '')
            text = text.replace(',','')
            txt.write(text + '\n')
    txt.close()
