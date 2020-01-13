# Project name: Segmentation of poor contrast grain boundaries in CT scan images (Master's thesis)
# Introduction
### What is grain boundary?
Most inorganic materials in nature (e.g. common metals, many ceramic, rocks) are polycrystalline structures, which are made of a large number of single crystalites (or grains). Also, the interface of two grains is called the grain boundary. The grain boundaries in a material's microstructure have significant effects on the material's pyhsical properties (e.g. electrical and thermal conductivity). For this reason, it is crucial to analyze the grain structure.
### What is CT scan?
The computed tomography (CT) scans are a fast qualitative diagnostic tool for producing 3D images of materials’ microstructure. However, the quantitave analyze is commonly necessary for scientific purposes. The quantitative analyse via CT scans might be possible by using image processing methods.
### What is the project?
The aim of the project is to segment the poor contrast grain boundaries in CT scan. You may see an example of a CT scan image in the figure below (1st picture) and its corresponding grain map (ground truth) at the 4th picture.
# Description
In this project, I implemented a convolutional neural networks structure, called the 3D U-Net, to enhance the contrast at grain boundaries of CT scan images (see https://arxiv.org/abs/1606.06650 for the details of the 3D U-Net). Then, I implemented a marker-based watershed segmentation algorithm* to complete the segmentation (see https://en.wikipedia.org/wiki/Watershed_(image_processing) for the details of watershed segmentation). In the figure below, you may see a slice of output of the 3D U-Net (2nd image) and output of the watershed algorithm (3rd image).

* I am not allowed to share the source code due to copyrights, please see https://iopscience.iop.org/article/10.1088/0965-0393/23/6/065001 for the details of algorithm.

You may see more results of 3D U-Net I employed in this project in the folder 'data'. The images are cutouts from the consecutive CT scan images and their corresponding output. Also, you may find the **weights of the 3D U-Net** I used on the link; https://cloudstore.uni-ulm.de/apps/files/?dir=/Shared/unet_thesis_weights&fileid=12379221#


![result](https://github.com/Iammuratc/grainDetection/blob/master/result.png)

 # Installation
 I used Python3 (e.g. Keras) and MATLAB for this project. Before the installation, please make sure you have downloaded the weights on the link provided above.
 
You can clone the repository by the command: 
```bash 
git clone git@github.com:Iammuratc/grainDetection.git
```
You may test the code on the images in the folder 'data'
```bash
python main3D.py
```
