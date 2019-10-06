from model import *
from data import *
from keras.utils import plot_model
import matplotlib.pyplot as plt

# from unet_master.testGenerator import timeSeriesGenerator, timeSeriesGenerator3D

import scipy.misc
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from write3Dimage import write3D
import os
import numpy as np
import scipy.io
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

netname='unet3D_weights.hdf5'

## Training

kernel=3
datasize=64
scale=2
# gen=timeSeriesGenerator3D(1,path,[20,39],[160,160,30], [760,760,600],target_size=(datasize,datasize,datasize),scale=scale)
# model = unet3D(input_size = (datasize,datasize,datasize,1),kernelSize=kernel)
# model.load_weights(netname)

# model.summary()
# model_checkpoint = ModelCheckpoint(netname, monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(gen,steps_per_epoch=300,epochs=20,callbacks=[model_checkpoint])



## Testing

data_path='C://Users//Murat//PycharmProjects//unet-thesis//thesis_data//low_temp//source'

mat='ct_normalized_t2.mat'
arrays = {}
f = h5py.File(os.path.join(data_path,mat))
a = list(f['normalized_imgs_backup'])
images= np.transpose(a) / (256.0*256.0-1)
print('input max: {}'.format(np.max(images[:,:,550])))

datasize=128

testimage=np.zeros(shape=(scale*datasize,scale*datasize,scale*datasize))

model = unet3D(input_size = (datasize,datasize,datasize,1),kernelSize=kernel)
model.load_weights(netname)
start=501

testimage=images[181:181+datasize*scale,181:181+datasize*scale,start:start+datasize*scale]

outpath=os.path.join(os.getcwd(),'data\\test_input')
write3D(testimage,outpath)

image=np.zeros(shape=(1,datasize,datasize,datasize,1))
image[0,:,:,:,0]=trans.resize(testimage,[datasize,datasize,datasize])
testimage=0

out=model.predict(image)
out=out[0,:,:,:,0]

outpath=os.path.join(os.getcwd(),'data\\test_output')
write3D(out, outpath, normalization=255.0)