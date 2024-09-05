import torch
import numpy as np
from torchvision import transforms
from scipy.ndimage import zoom
from PIL import Image as im
import cv2

# =======================
# Transforms the given image and label to pytorch tensor
# =======================
class ToTensor(object):

    def __call__(self, sample):
        # print('ToTensor start')
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']

            seg = np.array(seg)
            seg = torch.from_numpy(seg)
            seg = torch.unsqueeze(seg, 0).type(torch.float32)
        else:
            image, target = sample['image'], sample['target']

        image = np.array(image)
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0).type(torch.float32)
        
        target = np.array(target)
        target = torch.from_numpy(target)
        target = torch.unsqueeze(target, 0).type(torch.float32)

        if 'seg' in sample:
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Randomly translates the given image and label.
# Translation amounts in x and y axes are given as an array called translate.
# Translation in x axes is done by a value randomly chosen from (-translate[0], translate[0])
# Translation in y axes is done by a value randomly chosen from (-translate[1], translate[1])
# =======================
class Translate(object):
    
    def __init__(self, translate = None):
        self.translate = translate
    
    def __call__(self, sample):
        # print('Translate start')
        translation_x = np.random.randint(-self.translate[0], self.translate[0])
        translation_y = np.random.randint(-self.translate[1], self.translate[1])
        
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        image = np.expand_dims(cv2.warpAffine(image, M, (image.shape[1], image.shape[0])), axis = -1)
        target = np.expand_dims(cv2.warpAffine(target, M, (target.shape[1], target.shape[0])), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)

        if 'seg' in sample:
            seg = np.expand_dims(cv2.warpAffine(seg, M, (seg.shape[1], seg.shape[0])), axis = -1)
            seg = np.squeeze(seg)
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Randomly rotates the given image and label.
# Range for the rotation amount is given as an array called rotate
# and the angle is sampled from the range rotate[0] and rotate[1]
# =======================
class Rotate(object):
    
    def __init__(self, rotate = None):
        self.rotate = rotate
    
    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        angle = np.random.randint(self.rotate[0], self.rotate[1])
        
        image = np.expand_dims(np.array(im.fromarray(image).rotate(angle)), axis = -1)
        target = np.expand_dims(np.array(im.fromarray(target).rotate(angle)), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)
        
        if 'seg' in sample:
            seg = np.expand_dims(np.array(im.fromarray(seg).rotate(angle)), axis = -1)
            seg = np.squeeze(seg)   
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Randomly flips image and label around x or y axis with same probability
# =======================
class Flip(object):
    
    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        flip_axis = np.random.randint(0, 2)
        image = np.expand_dims(np.flip(image, axis = flip_axis), axis = -1)
        target = np.expand_dims(np.flip(target, axis = flip_axis), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)
        
        if 'seg' in sample:
            seg = np.expand_dims(np.flip(seg, axis = flip_axis), axis = -1)
            seg = np.squeeze(seg)
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Scales image and label using a random zoom_factor between [0.5, 1.5]
# If zoom_factor < 1.0 zoom_out
# If zoom_factor > 1.0 zoom_in
# =======================
# BUG -- NOT RETURNING 0-1 FOR BINARY IMAGES
# =======================
class ScaleByAFactor(object):
    
    def clipped_zoom(self, img, zoom_factor):

        h, w = img.shape[:2]
    
        # Zooming out
        if zoom_factor < 1:
    
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
    
            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_factor)
    
        # Zooming in
        elif zoom_factor > 1:
    
            out = zoom(img, zoom_factor)
            center_out_h, center_out_w = out.shape[0] // 2, out.shape[1] // 2
            center_img_h, center_img_w = h // 2, w // 2
            top_left_h = center_out_h - center_img_h
            top_left_w = center_out_w - center_img_w
            
            out = out[top_left_h:(top_left_h + h), top_left_w:(top_left_w + w)]
    
        # If zoom_factor == 1, just return the input array
        else:
            out = img
    
        return out
        
    def __call__(self, sample):
        # print('Scalebyafactor start')
        # image, target, seg = sample['image'], sample['target'], sample['seg']
        image, target = sample['image'], sample['target']

        scale = np.random.rand() + 0.5
        image = np.expand_dims(self.clipped_zoom(image, scale), axis = -1)
        target = np.expand_dims(self.clipped_zoom(target, scale), axis = -1)
        # seg = np.expand_dims(self.clipped_zoom(seg, scale), axis = -1)
        
        image = np.squeeze(image)
        target = np.squeeze(target)
        # seg = np.squeeze(seg)
        
        # return {'image': image, 'target': target, 'seg': seg}
        return {'image': image, 'target': target}

# =======================
# Randomly crop a region with a specified size from the image and label.
# Size should be specified as an array (output_size)
# where output_size[0] is the height and output_size[1] 
# is the width of the cropped image
# =======================
class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        new_h, new_w = self.output_size[0], self.output_size[1]

        h, w = image.shape[:2]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        target = target[top: top + new_h, left: left + new_w]
        
        if 'seg' in sample:
            seg = seg[top: top + new_h, left: left + new_w]
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Resize the given image and label to a specified size
# Size should be specified as an array (output_size)
# where output_size[0] is the height and output_size[1] 
# is the width of the resized image
# =======================
class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        new_h, new_w = self.output_size[0], self.output_size[1]

        new_h, new_w = int(new_h), int(new_w)
        
        tr = transforms.Compose([transforms.ToPILImage(), transforms.Resize((new_h, new_w)), transforms.ToTensor()])
        image = tr(image).squeeze().numpy()
        target = tr(target).squeeze().numpy()

        if 'seg' in sample:
            seg = (tr(seg).squeeze().numpy() > 0).astype(np.float32)
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Apply Color Jitter (brightness and contrast) to a given image
# =======================
class ColorJitter(object):    
    def __init__(self, brightness = (0.01, 2.0), contrast = (0.01, 2.0)):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']
        
        img = torch.from_numpy(image.copy()).float()
        pil_image = transforms.ToPILImage()(img)
        pil_tr = transforms.ColorJitter(brightness = self.brightness, contrast = self.contrast)(pil_image)
        image = np.asarray(pil_tr)

        if 'seg' in sample:
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}

# =======================
# Apply a set of transformations with some probability
# =======================
class RandomApply(object):
    
    def __init__(self, tf, p):
        self.tf = tf
        self.p = p
    
    def __call__(self, sample):
        
        if 'seg' in sample:
            image, target, seg = sample['image'], sample['target'], sample['seg']
        else:
            image, target = sample['image'], sample['target']

        p_sample = np.random.rand()
        if self.p > p_sample:
            sample = self.tf(sample)
            if 'seg' in sample:
                image, target, seg = sample['image'], sample['target'], sample['seg']
            else:
                image, target = sample['image'], sample['target']
        
        if 'seg' in sample:
            return {'image': image, 'target': target, 'seg': seg}
        else:
            return {'image': image, 'target': target}
