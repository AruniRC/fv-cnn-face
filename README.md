fv-cnn-face
===========

This repo is no longer maintained and I have discontinued this project (which was never properly completed). Code implementing Fisher Vector CNN will be linked here for completeness when/if the original authors make it available on their website.



D-CNN [1] on LFW Faces.

* train_gmm.m learns a Gaussian Mixture Model from 'conv5' features.
* fv_lfw_encoding.m  encodes each LFW image as a Fisher Vector using GMM created by train_gmm
* diagMetric_train.m trains a pseudo-diagonal metric similar to Fisher Vector Faces [2].

* Run on MATLAB R2014b.

* Part of the data is hosted on private UMass servers (
weibull: /scratch1/arunirc/FV_CNN_Face/data/all_img_lfw_funneled.mat)

* Please read the .txt files in the tools, models, cnn_models and data folders to obtain the proper data for this code to run.



1. Deep convolutional filter banks for texture recognition and segmentation. Mircea Cimpoi, Subhransu Maji, Andrea Vedaldi
ArXiv (Submitted on 25 Nov 2014).

2. Simonyan, Karen, et al. "Fisher vector faces in the wild." Proc. BMVC. Vol. 1. No. 2. 2013.
