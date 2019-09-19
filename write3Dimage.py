import scipy.misc
import numpy

def write3D(im, path, normalization=255.0):
    print(im.shape)
    ma=numpy.max(im)
    print(ma)
    im = im * normalization
    im = numpy.around(im)
    for z in range(0, im.shape[2]):
        #print(z)
        current_slice = im[:, :, z]
        scipy.misc.imsave(path + '\\slice_' + format(z, '04d')+'.tif', current_slice)
