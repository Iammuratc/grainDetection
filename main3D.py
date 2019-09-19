from unet_master.model import *
from unet_master.data import *
from keras.utils import plot_model
import matplotlib.pyplot as plt

from unet_master.testGenerator import timeSeriesGenerator, timeSeriesGenerator3D

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unet_master.write3Dimage import write3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path='D:\\Uni\\AlCu\\'

netname='unet3D.hdf5'
kernel=3
timesteps=1;
datasize=64-16
datasize=64
scale=2
gen=timeSeriesGenerator3D(1,path,[20,39],[160,160,30], [760,760,600],target_size=(datasize,datasize,datasize),scale=scale)
model = unet3D(input_size = (datasize,datasize,datasize,1),kernelSize=kernel)
model.load_weights(netname)

model.summary()
model_checkpoint = ModelCheckpoint(netname, monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(gen,steps_per_epoch=300,epochs=20,callbacks=[model_checkpoint])




time=65
datasize=128

#testimage=np.zeros(shape=(1,scale*datasize,scale*datasize,scale*datasize,1))
testimage=np.zeros(shape=(scale*datasize,scale*datasize,scale*datasize))
testpath='D:\\Daten_Projekte\\TessellationZeitreihe\\Data\\probe11_b__'
model = unet3D(input_size = (datasize,datasize,datasize,1),kernelSize=kernel)
model.load_weights(netname)
start=501
for t in range(start,start+datasize*scale):
    im = io.imread(
        path + 'probe11_b__' + format(time, '03d') + '\\0-ring_correction_cutout\\slice_' +
        format(t, '04d') + '.tif', as_gray=True) / 255.0
    tmp=np.zeros(shape=(scale*datasize,scale*datasize))
    tmp[0:np.shape(im)[0],0:np.shape(im)[1]]=im[181:181+datasize*scale,181:181+datasize*scale]
    #tmp=im[201:201+datasize,201:201+datasize]
    testimage[:,:,t-start]=tmp

image=np.zeros(shape=(1,datasize,datasize,datasize,1))
outpath='D:\\Uni\\AlCu\\probe11_b__045\\test2'
write3D(testimage,outpath)
image[0,:,:,:,0]=trans.resize(testimage,[datasize,datasize,datasize])
testimage=0

out=model.predict(image)
out=out[0,:,:,:,0]

fig,axs=plt.subplots(nrows=1,ncols=3, sharex=True, figsize=(3, 5))
axs[0].imshow(out[:,:,32],cmap='gray')
plt.show()

outpath='D:\\Uni\\AlCu\\probe11_b__045\\test'
write3D(out, outpath, normalization=255.0)

#
# for t in range(-timesteps,timesteps+1):
#     plt.figure()
#     plt.imshow(testimage[0,:,:,t+timesteps])
# plt.show()
