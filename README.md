# Hand-Eye Calibration in Unstructured Environments

In this project, we present a deep learning based method to perform hand-eye calibration in the wild. Our method
does not require any object of known geometry to be present in the scene. It estimates the hand-eye calibration
parameters by directly making sense of the visual features present in the images taken from different view-points.

For more details, visit the [project website](https://sites.google.com/andrew.cmu.edu/deep-hand-eye-calibration).

# Setup

To setup the environment

```
git clone https://github.com/mohith-sakthivel/deep_hand-eye_calib.git
cd deep_hand-eye_calib

conda env create -f environment.yaml
conda activate hand-eye
```


To download and setup the dataset

```
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip
unzip Rectified.zip -d data/DTU_MVS_2014

mv temp/camera_pose.json data/DTU_MVS_2014
```

To train the model

```
python -m deep_hand_eye.train
```