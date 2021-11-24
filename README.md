# **Human Action Recognition in Videos using Pedestrian Keypoints**
*Case study in Artificial Intelligence and Visual Computing as part of a [group school project](https://moodle.polytechnique.fr/course/view.php?id=13078)*.

___
___

- [**Introduction**](#introduction)
- [**The Dataset**](#the-dataset)
- [**Analysis and Visualization of the Dataset**](#analysis-and-visualization-of-the-dataset)
- [**Models**](#models)
- [**Resuts**](#results)
- [**Conclusion**](#conclusion)
- [**References**](#references)


___

## **Introduction**

Classifying pedestrians' every-day life actions is a state-of-the art topic in computer vision, including a lot of sub-topics like pedestrian detection, segmentation, video consolidation, classification, etc.

To recognize an action, a human being would localize the person first, analyze the position of its body-parts, see how these are moving and interacting,    and classify accordingly.

We propose here to reproduce this process, using the information of body-parts position (pedestrian keypoints/skeleton) along pedestrian videos.
The data to process is a sequence of labelled 3D coordinates on pedestrians, and the output is the label of an action (walking, running, biking, falling on the ground, fighting, etc.).

___

## **The Dataset**

The dataset used is the **NTU RGB+D - Action Recognition Dataset (Skeletons)** described in [this repository](https://github.com/shahroudy/NTURGB-D).
When cloning the current respository please download the following [zip file](https://drive.google.com/u/0/uc?export=download&confirm=7nHU&id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H) (5.8 Go) and extract it (13.4 Go) in the data folder. This compressed folder contains the relevant data for our project.

### ***Data Structure***

The *NTU RGB+D skeletons dataset* contains 56,880 skeletal data samples (3D locations of 25 major body joints at each frame).

302 samples have missing or incomplete skeleton data. The list of these are provided [here](./data/missing_skeletons.txt).

Each file in the dataset is in the format of SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013), in which sss is the setup number (between 0001 and 017), ccc is the camera ID, ppp is the performer (or subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

To know more about the setups, the camera IDs, and more details, please refer to the [NTU RGB+D dataset paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf).

___

## **Analysis and Visualization of the Dataset**

*TODO*

___

## **Models**

___

## **Results**

___

## **Conclusion**

___

### **References**

- [**NTU RGB+D - Action Recognition Dataset**](https://github.com/shahroudy/NTURGB-D)