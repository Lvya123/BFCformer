import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
from natsort import natsorted
import cv2
import math
from PIL import Image, ImageChops, ImageOps, ImageEnhance
from glob import glob
import copy
from einops import rearrange


# RAIN MASK AUGMENTATION CODE
# code modified from https://github.com/tsingqguo/efficientderain



# ROTATION DATA AUGMENTATION CODE
# Code modified from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

def get_translation_matrix_2d(dx, dy):
  """
  Returns a numpy affine transformation matrix for a 2D translation of
  (dx, dy)
  """
  return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotate_image(image, angle):
  """
  Rotates the given image about it's centre
  """

  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) / 2)

  rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
  trans_mat = np.identity(3)

  w2 = image_size[0] * 0.5
  h2 = image_size[1] * 0.5

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
  tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
  bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
  br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

  x_coords = [pt[0] for pt in [tl, tr, bl, br]]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in [tl, tr, bl, br]]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))
  new_image_size = (new_w, new_h)

  new_midx = new_w * 0.5
  new_midy = new_h * 0.5

  dx = int(new_midx - w2)
  dy = int(new_midy - h2)

  trans_mat = get_translation_matrix_2d(dx, dy)
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
  result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

  return result

def rotated_rect_with_max_area(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    # the other two corners are on the mid-line parallel to the longer line
    x = 0.5 * side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return int(wr), int(hr)

def gen_rotate_image(img, angle):
  dim = img.shape
  h = dim[0]
  w = dim[1]

  img = rotate_image(img, angle)
  dim_bb = img.shape
  h_bb = dim_bb[0]
  w_bb = dim_bb[1]

  w_r, h_r = rotated_rect_with_max_area(w, h, math.radians(angle))

  w_0 = (w_bb-w_r) // 2
  h_0 = (h_bb-h_r) // 2
  img = img[h_0:h_0 + h_r, w_0:w_0 + w_r, :]

  return img


# CODE FOR RAIN MASK AUGMENTATIONS
# code modified from https://github.com/tsingqguo/efficientderain

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(
      (pil_img.width, pil_img.height),
      Image.AFFINE, (1, level, 0, 0, 1, 0),
      resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(
      (pil_img.width, pil_img.height),
      Image.AFFINE, (1, 0, 0, level, 1, 0),
      resample=Image.BILINEAR)


def roll_x(pil_img, level):
  
  """Roll an image sideways."""
  delta = int_parameter(sample_level(level), pil_img.width / 3)
  if np.random.random() > 0.5:
    delta = -delta
  xsize, ysize = pil_img.size
  delta = delta % xsize
  if delta == 0: return pil_img
  part1 = pil_img.crop((0, 0, delta, ysize))
  part2 = pil_img.crop((delta, 0, xsize, ysize))
  pil_img.paste(part1, (xsize-delta, 0, xsize, ysize))
  pil_img.paste(part2, (0, 0, xsize-delta, ysize))

  return pil_img

def roll_y(pil_img, level):
  """Roll an image sideways."""
  delta = int_parameter(sample_level(level), pil_img.width / 3)
  if np.random.random() > 0.5:
    delta = -delta
  xsize, ysize = pil_img.size
  delta = delta % ysize
  if delta == 0: return pil_img
  part1 = pil_img.crop((0, 0, xsize, delta))
  part2 = pil_img.crop((0, delta, xsize, ysize))
  pil_img.paste(part1, (0, ysize-delta, xsize, ysize))
  pil_img.paste(part2, (0, 0, xsize, ysize-delta))

  return pil_img

# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
  level = float_parameter(sample_level(level), 1.8) + 0.1
  return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
  level = float_parameter(sample_level(level), 1.8) + 0.1
  return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
  level = float_parameter(sample_level(level), 1.8) + 0.1
  return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
  level = float_parameter(sample_level(level), 1.8) + 0.1
  return ImageEnhance.Sharpness(pil_img).enhance(level)

def zoom_x(pil_img, level):
  # zoom from .02 to 2.5
  rate = level
  zoom_img = pil_img.transform(
      (pil_img.width, pil_img.height),
      Image.AFFINE, (rate, 0, 0, 0, 1, 0),
      resample=Image.BILINEAR)
  # need to do reflect padding
  if rate > 1.0:
    orig_x, orig_y = pil_img.size
    new_x = int(orig_x/rate)
    zoom_img = np.array(zoom_img)
    zoom_img = np.pad(zoom_img[:, :new_x, :], ((0, 0), (0, orig_x-new_x), (0,0)), 'wrap')
  return zoom_img

def zoom_y(pil_img, level):
  # zoom from .02 to 2.5
  rate = level
  zoom_img = pil_img.transform(
      (pil_img.width, pil_img.height),
      Image.AFFINE, (1, 0, 0, 0, rate, 0),
      resample=Image.BILINEAR)
  # need to do reflect padding
  if rate > 1.0:
    orig_x, orig_y = pil_img.size
    new_y = int(orig_y/rate)
    zoom_img = np.array(zoom_img)
    zoom_img = np.pad(zoom_img[:new_y, :, :], ((0, orig_y-new_y), (0, 0), (0,0)), 'wrap')
  return zoom_img
    

augmentations = [
    rotate, shear_x, shear_y,
    zoom_x, zoom_y, roll_x, roll_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    roll_x, roll_y, color, contrast, brightness, sharpness
]


# RAIN MASK AUGMENTATION CODE
# code modified from https://github.com/tsingqguo/efficientderain

class RandomCrop(object):
  def __init__(self, image_size, crop_size):
    self.ch, self.cw = crop_size
    ih, iw = image_size

    self.h1 = random.randint(0, ih - self.ch)
    self.w1 = random.randint(0, iw - self.cw)

    self.h2 = self.h1 + self.ch
    self.w2 = self.w1 + self.cw

  def __call__(self, img):
    if len(img.shape) == 3:
      return img[self.h1: self.h2, self.w1: self.w2, :]
    else:
      return img[self.h1: self.h2, self.w1: self.w2]

def getRainLayer2(rand_id1, rand_id2, rain_mask_dir):
  path_img_rainlayer_src = os.path.join(rain_mask_dir, f'{rand_id1}-{rand_id2}.png')
  rainlayer_rand = cv2.imread(path_img_rainlayer_src).astype(np.float32) / 255.0
  rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
  return rainlayer_rand

def getRandRainLayer2(rain_mask_dir):
  rand_id1 = random.randint(1, 165)
  rand_id2 = random.randint(4, 8)
  rainlayer_rand = getRainLayer2(rand_id1, rand_id2, rain_mask_dir)
  return rainlayer_rand
  
def rain_aug(img_rainy, img_gt, rain_mask_dir, zoom_min = 0.06, zoom_max = 1.8):
  img_rainy = (img_rainy.astype(np.float32)) / 255.0
  img_gt = (img_gt.astype(np.float32)) / 255.0
  img_rainy_ret = img_rainy
  img_gt_ret = img_gt

  rainlayer_rand2 = getRandRainLayer2(rain_mask_dir)
  rainlayer_aug2 = augment_and_mix(rainlayer_rand2, severity = 3, width = 3, depth = -1, zoom_min = zoom_min, zoom_max = zoom_max) * 1

  height = min(img_rainy.shape[0], rainlayer_aug2.shape[0])
  width = min(img_rainy.shape[1], rainlayer_aug2.shape[1])
  
  cropper = RandomCrop(rainlayer_aug2.shape[:2], (height, width))
  rainlayer_aug2_crop = cropper(rainlayer_aug2)
  cropper = RandomCrop(img_rainy.shape[:2], (height, width))
  img_rainy_ret = cropper(img_rainy_ret)
  img_gt_ret = cropper(img_gt_ret)
  img_rainy_ret = img_rainy_ret + rainlayer_aug2_crop - img_rainy_ret*rainlayer_aug2_crop
  img_rainy_ret = np.clip(img_rainy_ret, 0.0, 1.0)
  img_rainy_ret = (img_rainy_ret * 255).astype(np.uint8)
  img_gt_ret = (img_gt_ret * 255).astype(np.uint8)
    
  return img_rainy_ret, img_gt_ret

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1., zoom_min=0.06, zoom_max=1.8):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image)
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(2, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations)
      if (op == zoom_x or op == zoom_y):
        rate = np.random.uniform(low=zoom_min, high=zoom_max)
        image_aug = apply_op(image_aug, op, rate)
      else:
        image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * image_aug
    
  max_ws = max(ws)
  rate = 1.0 / max_ws  
  
  mixed = max((1 - m), 0.7) * image + max(m, rate*0.5) * mix
  return mixed

def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) / 255.




# DataLoaders for Training and Validation set

class GTRainDataset(Dataset):
  """
    The dataset class for weather net training and validation.

    Parameters:
        train_dir_list (list) -- list of dirs for the dataset.
        val_dir_list (list) -- list of dirs for the dataset.
        rain_mask_dir (string) -- location of rain masks for data augmentation.
        img_size (int) -- size of the images after cropping.
        is_train (bool) -- True for training set.
        val_list (list) -- list of validation scenes
        sigma (int) -- variance for random angle rotation data augmentation
        zoom_min (float) -- minimum zoom for RainMix data augmentation
        zoom_max (float) -- maximum zoom for RainMix data augmentation
  """
  def __init__(self, train_dir_list, val_dir_list, rain_mask_dir, img_size, is_train=True, sigma=13, zoom_min=0.06, zoom_max=1.8):
    super(GTRainDataset, self).__init__()

    self.rain_mask_dir = rain_mask_dir
    self.img_size = img_size
    self.is_train = is_train
    self.img_paths = []
    self.sigma = sigma
    self.zoom_min = zoom_min
    self.zoom_max = zoom_max
    self.scene_indices = []
    last_index = 0
    scene_paths = []
    train_dir_list = [os.path.join(train_dir_list,d) for d in os.listdir(train_dir_list)]
    val_dir_list = [os.path.join(val_dir_list,d) for d in os.listdir(val_dir_list)]

    if is_train:
      root_dir_list = train_dir_list
    else:
      root_dir_list = val_dir_list
    
   
    for root_dir in root_dir_list:
      # print('root_dir =',root_dir)
      # print('root_dir_list =',root_dir_list)
      # print(list(os.walk(root_dir))[0])

      # scene_paths += [os.path.join(root_dir, scene) for scene in list(os.walk(root_dir))[0][1]]
      scene_paths += [os.path.join(root_dir, scene) for scene in os.listdir(root_dir)]
    
    for scene_path in root_dir_list:
      # print(scene_path)
      scene_img_paths = natsorted(glob(os.path.join(scene_path, '*R-*.png')))

      scene_length = len(scene_img_paths)
      self.scene_indices.append(list(range(last_index, last_index + scene_length)))
      last_index += scene_length
      self.img_paths += scene_img_paths
    
    # number of images in full dataset
    self.data_len = len(self.img_paths)
  
  def __len__(self):
    return self.data_len

  def get_scene_indices(self):
    return self.scene_indices
  
  def __getitem__(self, index):
    ts = self.img_size

    inp_path = self.img_paths[index]
    tar_path = self.img_paths[index][:-9] + 'C-000.png'
    if('Gurutto_1-2' in inp_path):
      tar_path = self.img_paths[index][:-9] + 'C' + self.img_paths[index][-8:]
    inp_img = Image.open(inp_path)
    tar_img = Image.open(tar_path)

    # print(tar_path)

    # To numpy
    inp_img = np.array(inp_img)
    tar_img = np.array(tar_img)

    if self.is_train:
      if random.randint(1, 10) > 4:
        inp_img, tar_img = rain_aug(inp_img, tar_img, self.rain_mask_dir, zoom_min=self.zoom_min, zoom_max=self.zoom_max)

    # Random rotation
    if self.is_train:
      angle = np.random.normal(0, self.sigma)
      inp_img_rot = gen_rotate_image(inp_img, angle)
      if (inp_img_rot.shape[0] >= 256 and inp_img_rot.shape[1] >= 256):
        inp_img = inp_img_rot
        tar_img = gen_rotate_image(tar_img, angle)

    # reflect pad and random cropping to ensure the right image size for training
    h,w = inp_img.shape[:2]

    # To tensor
    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    # print('inp_path = ', inp_path)

    # reflect padding
    # print(inp_img.shape)
    # print(type(ts))
    # print(type(w))
    padw = ts-w if w<ts else 0
    padh = ts-h if h<ts else 0

    # print('init inp_img.shape = ', inp_img.shape)
    # print('padw = ', padw)
    # print('padh = ', padh)
    # print('inp_path = ', inp_path)
    # print('len(inp_img) = ', len(inp_img))

    if padw!=0 or padh!=0:
      # if (len(inp_img) == 4):
      #   inp_img._squeeze(0)
      inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='constant')
      tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='constant')

    # print('pad inp_img.shape = ', inp_img.shape)

    # print('pad inp_img.shape = ', inp_img.shape)

     
    if self.is_train:
      # random cropping
      hh, ww, = inp_img.shape[1], inp_img.shape[2]
      rr = random.randint(0, abs(hh-ts))
      cc = random.randint(0, abs(ww-ts))
      inp_img = inp_img[:, rr:rr+ts, cc:cc+ts]
      tar_img = tar_img[:, rr:rr+ts, cc:cc+ts]
      # print(inp_img.shape)
    else:
      # center cropping
      inp_img = TF.center_crop(inp_img, (ts, ts))
      tar_img = TF.center_crop(tar_img, (ts, ts))

    # print('inp_img.shape = ', inp_img.shape)

    # Data augmentations: flip x, flip y
    if self.is_train:
      aug = random.randint(0, 2)
    else:
      aug = 0
    
    if aug==1:
      inp_img = inp_img.flip(1)
      tar_img = tar_img.flip(1)
    elif aug==2:
      inp_img = inp_img.flip(2)
      tar_img = tar_img.flip(2)

    # Get image name
    file_name = inp_path.split('/')[-1]

    # Dict for return
    # If using tanh as the last layer, the range should be [-1, 1]
    sample_dict = {
        'input_img': inp_img,
        'target_img': tar_img,
        'file_name': file_name
    }

    return sample_dict
  


  # Samplper for the images

class CustomBatchSampler():
  def __init__(self, scene_indices, batch_size=16):
    self.scene_indices = scene_indices
    self.batch_size = batch_size
    self.num_batches = int(scene_indices[-1][-1]/batch_size)

  def __len__(self):
    return self.num_batches

  def __iter__(self):
    scene_indices = copy.deepcopy(self.scene_indices)
    for scene_list in scene_indices:
      random.shuffle(scene_list)
    out_indices = []
    done = False
    while not done:
      out_batch_indices = []
      if (len(scene_indices) < self.batch_size):
        self.num_batches = len(out_indices)
        return iter(out_indices)
      chosen_scenes = np.random.choice(len(scene_indices), self.batch_size, replace = False)
      empty_indices = []
      for i in chosen_scenes:
        scene_list = scene_indices[i]
        out_batch_indices.append(scene_list.pop())
        if (len(scene_list) == 0):
          empty_indices.append(i)
      empty_indices.sort(reverse=True)
      for i in empty_indices:
        scene_indices.pop(i)
      out_indices.append(out_batch_indices)
    self.num_batches = len(out_indices)
    return iter(out_indices)