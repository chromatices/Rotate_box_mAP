import os
import json

json_file = "coco_instances_results(rotated-faster_rcnn_R_50_FPN_3x).json"  # -> detectron format

dir_name = 'input\\rotated-detection-results'


try:
    if not (os.path.isdir(dir_name)):
        os.makedirs(os.path.join(dir_name))
except OSError as e:
    if e.errno != e.errno.EXIST:
        print("Failed to create directory.")
        raise

with open(json_file) as f:
    detectron_json = json.load(f)



image_ids = []

categories = {"Upper_molar": 0, "Upper_premolar": 1, "Upper_Canine": 2,
              "Upper_Lateral_incisor": 3, "Upper_Central_incisor": 4,
              "Lower_molar": 5, "Lower_premolar": 6, "Lower_Canine": 7,
              "Lower_Lateral_incisor": 8, "Lower_Central_incisor": 9,
              'Implant': 10}

categories = dict(zip(categories.values(), categories.keys()))

for i in range(len(detectron_json)):
    image_ids.append(detectron_json[i]['image_id'])

image_ids = list(set(image_ids))

for j in range(len(image_ids)):
    txt_name = str(image_ids[j]) + ".txt"
    txt = open(dir_name + '/' + txt_name, 'w')
    for i in range(len(detectron_json)):
        text = ""
        if image_ids[j] == detectron_json[i]['image_id']:
            text += categories.get(detectron_json[i]['category_id']) + " " + str(detectron_json[i]['score'])
            text += " " + str(detectron_json[i]['bbox'])
            text = text.replace('[', '')
            text = text.replace(']', '')
            text = text.replace(',', '')
            txt.write(text + '\n')
    txt.close()
