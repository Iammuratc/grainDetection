import skimage.io as io
import os
from random import randint
import scipy
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans


def timeSeriesGenerator3D(batch_size,path,time_area,image_low,image_high,target_size=(256,256),scale=1,seed=1,normalization=255.0):

    lowTime=time_area[0]
    highTime=time_area[1]
    low= image_low #+ [scipy.math.ceil(target_size / 2),time_steps]

   # high=np.array(image_high)- np.array([target_size[0],target_size[1], 2*time_steps+1])
 #   print(np.array(image_high))
#    print(np.array([target_size[0],target_size[1], target_size[2]]))
    high=np.array(image_high)- np.array([scale*target_size[0],scale*target_size[1], scale*target_size[2]])


    while True:
        input=np.zeros(shape=(batch_size,target_size[0],target_size[1],target_size[2],1))
        labels=np.zeros(shape=(batch_size,target_size[0],target_size[1],target_size[2],1))
        for i in range(0, batch_size ):
            currentInput=np.zeros(shape=(scale*target_size[0],scale*target_size[1],scale*target_size[2]))
            currentLabels=np.zeros(shape=(scale*target_size[0],scale*target_size[1],scale*target_size[2]))

            t=randint(lowTime, lowTime)
            corner=[randint(low[0],high[0]),randint(low[1],high[1]),randint(low[2],high[2])]



            for z in range(0,scale*target_size[2]):
                im = io.imread(
                    path + 'probe11_b__' + format(t, '03d') + '\\0-ring_correction_cutout\\slice_' +
                    format(corner[2]+z, '04d') + '.tif', as_gray=True) / normalization
                labelLarge = io.imread(
                    path + 'probe11_b__' + format(t, '03d') + '\\3-regions\\slice_' +
                    format(corner[2]+z, '04d') + '.tif', as_gray=True) / normalization
                currentInput[:,:,z]=im[corner[0]: corner[0] + scale*target_size[0], corner[1]: corner[1] +  scale* target_size[1]]
                currentLabels[:,:,z]=labelLarge[corner[0]: corner[0] + scale*target_size[0], corner[1]: corner[1] +scale* target_size[1]]
                #input[i, :, :, z,0] = im[corner[0]: corner[0] + target_size[0], corner[1]: corner[1] + target_size[1]]
                #labels[i,:,:,z,0]= labelLarge[corner[0]: corner[0] + target_size[0], corner[1]: corner[1] + target_size[1]]
            currentInput=trans.resize(currentInput,target_size)
            currentLabels=trans.resize(currentLabels,target_size)
            input[i,:,:,:,0]=currentInput
            labels[i,:,:,:,0]=currentLabels


        yield input,labels#(images,label)
