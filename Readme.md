# These files was created during my student job as a HiWi. Please note that these are the rough scripts, while originals are not provided due to NDA.

## CSV Corrector
- File name CSV was given from the .csv files.
- Purpose of this was to correct the laser printer angle which was directly connected to a .csv file, where x and y coordinates were provided.

## Controller.cpp or Controller.py
- This is used to control the motion of camera according to the user's will. The camera used is a 360 degree gimbal.
- Both the cpp and py files serves similar purpose

## LIFT.py
- This is used to detect the quality of laser printed polymer.
- First, a manual annotation was done on training images which were later provided to use on test images. Hough Transform is being used, as the polymer prints are in circular geometry.

## VAE_4.py
- This is a Variational Autoencoder, it is carried out from a standard template, however, multiple changes were carried out.
- The parameters played a crucial role, which allowed the VAE to work efficiently that it provided us with 100% accurate results. Please note that, pre-processing of images were carried out.

## VisioPro
- An alternate to VAE_4, this was created from scratch to detect any part of an complete image for the purpose of tracking the original image/part.

## Detection.cpp
- This file was created for the purpose of Object detection using pre-trained YOLO model.

## vcs_bit.cpp
- It is a general Version Control System Bit.



