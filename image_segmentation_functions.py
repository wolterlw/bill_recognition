import numpy as np
import cv2
from keras.models import load_model

class Segmenter(object):
    """methods needed to segment images"""
    def __init__(self):
        super(Segmenter, self).__init__()
        self.model = load_model('../trained/unet_model.hdf5')

    def _get_hline(self, pts):
        lines = np.hstack([pts, np.roll(pts, 1, axis=0)])
        hidx = np.argmin(np.abs(lines[:, 1] - lines[:, 3]))
        hline = lines[hidx].reshape(2,2)
        vec = hline[0] - hline[1]
        if vec[0] < 0:
            vec = - vec
        return vec if vec[0] > 0 else -vec

    def _crop_to_mask(self, x, y, cropped_width=300):
        img = (y>0.9).astype('uint8')*255

        img, contours, _ = cv2.findContours(img, 1, 2)
        cnt = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(rect)
        x_c, y_c = rect[0]

        vec = self._get_hline(points)
        angle = np.arctan(vec[1]/vec[0]) / np.pi * 180
        
        rows, cols, _ = x.shape
        center_big = tuple( np.r_[x_c,y_c]*x.shape[0]/y.shape[0] )

        M = cv2.getRotationMatrix2D(center_big,angle,1)
        
        dsize = x.shape[:2][::-1]
        x_img = cv2.warpAffine(x,M,dsize)
        #resizing the mask to fit the resized input image
        mask = cv2.resize(img/255,dsize)
        mask = cv2.warpAffine(mask[:,:,np.newaxis], M, dsize)

        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,25)),iterations=5)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5)),iterations=5)

        x_img = x_img[:,:,0]*mask

        x_min = min(np.argwhere(x_img.sum(axis=0)>0))[0]
        x_max = max(np.argwhere(x_img.sum(axis=0)>0))[0]
        y_min = min(np.argwhere(x_img.sum(axis=1)>0))[0]
        y_max = max(np.argwhere(x_img.sum(axis=1)>0))[0]

        x_crop = x_img[y_min:y_max,x_min:x_max]
        mask_crop = mask[y_min:y_max,x_min:x_max]
        x_crop[x_crop < 5] = 0
        
        dsize = (int(x_crop.shape[0] * cropped_width / x_crop.shape[1]), cropped_width)
        return {
            'rotated': x_img,
            'cropped': cv2.resize(x_crop, dsize[::-1]), 
            'mask': cv2.resize(mask_crop, dsize[::-1]),
            'center': center_big,
            'angle': angle,
            'heigth': y_max - y_min,
            'width': x_max - x_min
            }

    def _get_mask(self, img):
        x_inp = cv2.resize(img,(240,320)).astype('float32') / 255
        mask = self.model.predict(
            [x_inp.reshape(1,320,240,3)],
            verbose = False)[0,:,:,0]
        return mask

    def process_image(self, img, return_coord = False):

        inp_img = cv2.resize(
            img, (960,1280)
        )

        mask = self._get_mask(inp_img)
        cropped_dict = self._crop_to_mask(inp_img, mask, cropped_width=400)

        if return_coord:
            return cropped_dict
        else:
            return (cropped_dict['cropped'], cropped_dict['mask'])





            
