### This repository was created by combining Cartucho's mAP with 
### the R2CNN-Plus-Plus_Tensorflow/blob/master/libs/box_utils/iou_rotate.pystore in DetectionTeamUCAS.


### [Cartucho's mAP](https://github.com/Cartucho/mAP)

 
#### [DetectionTeamUCAS's R2CNN-Plus_Plus_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow/blob/master/libs/box_utils/iou_rotate.py)


## RBBox mAP(Rotated Bounding Box mean Average Precision)

[![GitHub stars](https://ifh.cc/g/9UgFG.gif)](https://github.com/chromatices/Rotate_box_mAP)
## HOW TO USE


## You should have to build cython file. Do first this shell
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

#### If you want other information, please visit Cartucho's mAP repository.
#### There has many Reference like IoU and other things.


## Authors:
* **chromatices** - Please give me your feedback: to. bomebug15@ds.seoultech.ac.kr
