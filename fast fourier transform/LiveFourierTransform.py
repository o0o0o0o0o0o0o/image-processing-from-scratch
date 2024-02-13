import numpy as np
import time
import cv2

def LiveFFT(rows,columns,numShots = 1e5,nContrIms = 30):
    imMin = .004  # minimum allowed value of any pixel of the captured image
    contrast = np.concatenate((
        np.zeros((10, 1)), np.ones((10, 1))), axis=1)  # internal use.

    window_name = 'Livefft by lixin'

    vc = cv2.VideoCapture(0)  # camera device
    cv2.namedWindow(window_name, 0)  # 0 makes it work a bit better
    cv2.resizeWindow(window_name, 1024, 768)  # this doesn't keep
    rval, frame = vc.read()
    # we need to wait a bit before we get decent images
    print("warming up camera... (.1s)")
    time.sleep(.1)
    rval = vc.grab()
    rval, frame = vc.retrieve()
    # determine if we are not asking too much
    frameShape = np.shape(frame)
    if rows > frameShape[1]:
        rows = frameShape[1]
    if columns > frameShape[0]:
        columns = frameShape[0]

    # calculate crop
    vCrop = np.array([np.ceil(frameShape[0] / 2. - columns / 2.),
                           np.floor(frameShape[0] / 2. + columns / 2.)], dtype=int)
    hCrop = np.array([np.ceil(frameShape[1] / 2. - rows / 2.),
                           np.floor(frameShape[1] / 2. + rows / 2.)], dtype=int)
    # start image cleanup with something like this:
    # for a running contrast of nContrIms frames
    contrast = np.concatenate((
        np.zeros((nContrIms, 1)),
        np.ones((nContrIms, 1))),
        axis=1)

    Nr = 0
    # main loop
    while Nr <= numShots:
        a = time.time()
        Nr += 1
        contrast = work_func(vCrop,hCrop,vc,imMin,window_name,contrast)
        print('framerate = {} fps \r'.format(1. / (time.time() - a)))
    # stop camera
    vc.release()

def work_func(vCrop,hCrop,vc,imMin,figid,contrast):
    # read image
    rval = vc.grab()
    rval, im = vc.retrieve()
    im = np.array(im, dtype=float)

    # crop image
    im = im[vCrop[0]: vCrop[1], hCrop[0]: hCrop[1], :]

    # pyramid downscaling
    # im = cv2.pyrDown(im)

    # reduce dimensionality
    im = np.mean(im, axis=2, dtype=float)
    # make sure we have no zeros
    im = (im - im.min()) / (im.max() - im.min())
    im = np.maximum(im, imMin)
    Intensity = np.abs(np.fft.fftshift(np.fft.fft2(im))) ** 2

    Intensity += imMin

    # kill the center lines for higher dynamic range
    # by copying the next row/column
    # h, w = np.shape(Intensity)
    # Intensity[(h / 2 - 1):(h / 2 + 1), :] = Intensity[(h / 2 + 1):(h / 2 + 3), :]
    # Intensity[:, (w / 2 - 1):(w / 2 + 1)] = Intensity[:, (w / 2 + 1):(w / 2 + 3)]

    # running average of contrast
    ##circshift contrast matrix up
    contrast = contrast[np.arange(1, np.size(contrast, 0) + 1) % np.size(contrast, 0), :]
    ##replace bottom values with new values for minimum and maximum
    contrast[-1, :] = [np.min(Intensity), np.max(Intensity)]

    maxContrast = 1
    minContrast = 7   # to be modify
    # openCV draw
    vmin = np.log(contrast[:, 0].mean()) + minContrast
    vmax = np.log(contrast[:, 1].mean()) - maxContrast
    Intensity = (np.log(Intensity + imMin) - vmin) / (vmax - vmin)
    Intensity = Intensity.clip(0., 1.)
    # Intensity = (Intensity - Intensity.min()) / (Intensity.max() - Intensity.min())

    time.sleep(.01)
    cv2.imshow(figid, np.concatenate((im, Intensity), axis=1))

    cv2.waitKey(1)

    return contrast


if __name__ == '__main__':
    LiveFFT(400,400)