import skimage.io as io
import os
from random import randint
import scipy
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans




def timeSeriesGenerator(batch_size,path,time_area,image_low,image_high, time_steps=1,target_size=(256,256),seed=1,normalization=255.0):

    lowTime=time_area[0];
    highTime=time_area[1]-2*time_steps-1;
    low= image_low #+ [scipy.math.ceil(target_size / 2),time_steps]

   # high=np.array(image_high)- np.array([target_size[0],target_size[1], 2*time_steps+1])
    high=np.array(image_high)- np.array([target_size[0],target_size[1], 0])



    #for i in range(1, batch_size+1):
    while True:
        input=np.zeros(shape=(batch_size,target_size[0],target_size[1],2*time_steps+1))
        labels=np.zeros(shape=(batch_size,target_size[0],target_size[1],1))
        for i in range(0, batch_size ):
            t=randint(lowTime, lowTime)
            corner=[randint(low[0],high[0]),randint(low[1],high[1]),randint(low[2],high[2])]


            label=np.zeros(shape=(target_size[0],target_size[1]))
            for time in range(0, 2*time_steps+1):
                im = io.imread(
                    path + 'probe11_b__' + format(t+time, '03d') + '\\0-ring_correction_cutout\\slice_' +
                    format(corner[2],'04d') + '.tif',as_gray=True)/normalization

                input[i,:,:,time]=im[corner[0]: corner[0]+target_size[0],corner[1]: corner[1]+target_size[1] ]

            labelLarge=io.imread(
                    path + 'probe11_b__' + format(t+time, '03d') + '\\3-regions\\slice_' +
                    format(corner[2],'04d') + '.tif',as_gray=True)/normalization
            label=labelLarge[corner[0]: corner[0]+target_size[0],corner[1]: corner[1]+target_size[1] ]
            #out[0:time_steps,i,:,:]=images
            labels[i,:,:,0]=label


        #print('shape')
        #print(shape((out,labels)))
        yield input,labels#(images,label)

def timeSeriesGenerator3D(batch_size,path,time_area,image_low,image_high,target_size=(256,256),scale=1,seed=1,normalization=255.0):

    lowTime=time_area[0];
    highTime=time_area[1];
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
#
#
# path='D:\\Daten_Projekte\\TessellationZeitreihe\\Data\\'
# time_area=[20,75]
# time_steps=1
# lowTime=time_area[0]+time_steps;
# highTime=time_area[1]-time_steps;
# print(path+'probe11_b__'+format(lowTime, '03d')+'\\0-ring_correction_cutout\\slice_'+format(74,'04d')+ '.tif' )
# im=io.imread(path+'probe11_b__'+format(lowTime, '03d')+'\\0-ring_correction_cutout\\slice_'+format(74,'04d')+ '.tif',as_gray = True)
# gen=timeSeriesGenerator(5,path,[20,39],[160,160,30], [660,660,400])
# for i in range(1, 10+1):
#     print(i)
#     X,Y=gen.__next__()
#     print(np.shape(X))
# fig,axs=plt.subplots(nrows=2,ncols=2, sharex=True, figsize=(3, 5))
# axs[0][0].imshow(X[0][:,:,0],cmap='gray')
# axs[0][1].imshow(X[0][:,:,1],cmap='gray')
# axs[1][0].imshow(X[0][:,:,2],cmap='gray')
# axs[1][1].imshow(X[1],cmap='gray')
# plt.figure()
# plt.imshow(X[0][:,:,2],cmap='gray')
# plt.show()
