#### This repository was created by combining Cartucho's mAP with 
#### the R2CNN-Plus-Plus_Tensorflow/blob/master/libs/box_utils/iou_rotate.py store in DetectionTeamUCAS.


### * [Cartucho's mAP](https://github.com/Cartucho/mAP)
### * [DetectionTeamUCAS's R2CNN-Plus_Plus_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow/blob/master/libs/box_utils/iou_rotate.py)


### RBBox mAP(Rotated Bounding Box mean Average Precision)   
- rotated output   
![9](https://user-images.githubusercontent.com/56014940/116551810-40c4d600-a933-11eb-85e5-4ec82d57b681.jpg)   
- horizontal output   
![9](https://user-images.githubusercontent.com/56014940/116551849-4d492e80-a933-11eb-9769-d3745e1b89d4.jpg)   
- rotated ground truth on horizontal predict output
![9](https://user-images.githubusercontent.com/56014940/116551916-62be5880-a933-11eb-9321-592cc70d6559.jpg)



### This repository did not use exact coordinates, and used data to convert from the existing bbox coordinates to the Rbbox coordinates to ensure that the code returns normally.

## HOW TO USE


## You should have to build cython file if you want to using GPU on calculating IoU. You don't have to if you using cpu.
```
$ python setup.py build_ext --inplace
```
#### use_gpu
```
$ python main.py --gpu gpu
```
#### use_cpu
```
$ python main.py
```
### IF YOU USING ON GOOGLE COLAB, MUST HAVE TO
```
$ python main.py -na --gpu (cpu,gpu)
```
Animator does not work on colab.

### Bounding Box Mode example

Box mode has default value as 'rotated'. So if you want measure on horizontal prediction, you should have to give each box mode

```
$ python main.py --gt-box-mode horizontal --pred-box-mode horizontal
```
#### If you want other information of mean Average Precision, please visit [Cartucho's mAP](https://github.com/Cartucho/mAP) repository.
#### There has many Reference like IoU and other things.


## Authors:
* **chromatices** - Please give me your feedback: to. bomebug15@ds.seoultech.ac.kr
