import numpy as np
import matplotlib.pyplot as plt
import imageio

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

def draw_Gaussian_2d(im, x0, y0, sigma_X, sigma_Y, theta, A):
  X, Y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))

  a = np.cos(theta)**2/(2*sigma_X**2) + np.sin(theta)**2/(2*sigma_Y**2)
  b = -np.sin(2*theta)/(4*sigma_X**2) + np.sin(2*theta)/(4*sigma_Y**2)
  c = np.sin(theta)**2/(2*sigma_X**2) + np.cos(theta)**2/(2*sigma_Y**2)

  Z = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
  
  if(len(im.shape) == 2):
    return im + Z
  elif(len(im.shape) == 3):
    return im + np.repeat(Z[:,:,np.newaxis], im.shape[2], axis=2)
  else:
    raise ValueError("Image must be 2D or 3D")

def gen_multi_max(noise):
  im = np.zeros((100, 100))
  im = draw_Gaussian_2d(im, 30, 40, 18, 25, -0.3, 1.7)
  im = draw_Gaussian_2d(im, 20, 90, 5, 5, 0, 2)
  im = draw_Gaussian_2d(im, 80, 30, 12, 5, 1.8, 1.9)
  im = draw_Gaussian_2d(im, 80, 80, 15, 15, 0, 1.6)

  im = im / im.max()
  if noise > 0:
    im = im + np.random.randn(*im.shape) * noise
    im = im / im.max()
  return im


def noise_image(image, noise_typ):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def normalize_image(image, value_range):
    return (image / (value_range - 1)).astype(np.float64)
  
def denormalize_image(image, value_range):
    return (image * (value_range - 1)).astype(np.int32)

def invert_image(image):
    return 1 - image

def crop_image(image, y_s, y_e, x_s, x_e):
    return image[y_s:y_e, x_s:x_e,:]

def get_dims(image, i):
  if(isinstance(i, list)):
    return np.take(image, i, 2)
  else:
    return np.expand_dims(image[:,:,i], axis=2)

def get_image(path):
    image = imageio.imread(path)
    
    if(image.ndim == 2):
        return np.expand_dims(image, axis=2)
    elif(image.ndim == 3):
        return np.array(image)
    else:
        raise ValueError("The image must be 2 or 3 dimensional.")

def show_image(image):
  if(len(image.shape) == 2 or image.shape[2] == 1):
    imshow(image, cmap="gray")
  else:
    if(image.shape[2] == 3):
      imshow(image, cmap="gray")
    for i in range(image.shape[2]):
      imshow(image[:,:,i], cmap="gray")

def sc_show_image(image, sc_image):
  imshow(image, cmap="gray")
  imshow(sc_image, cmap="gray")


def __num_to_binary_list(number, bit_range):
  return [(number >> bit) & 1 for bit in range(bit_range - 1, -1, -1)]

def __binary_list_to_num(bin_lst):
  out = 0
  for bit in bin_lst:
    out = (out << 1) | bit
  return out

def __linearize_list(lst):
  lst_out = []
  for x in range(len(lst[0])-1,-1,-1):
    for y in range(len(lst)):
      lst_out.append(lst[y][x])
  return lst_out[::-1]

def __delinearize_list(lst, channel_num):
  lst_out = [[] for y in range(channel_num)]
  for i in range(len(lst)):
    lst_out[i % channel_num].append(lst[i])
  return lst_out[::-1]

def to_single_channel(image, bit_range=8):
  image_out = np.empty(image.shape[:2], dtype=np.int32)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      binary_channels = []
      for c in range(image.shape[2]):
        binary_channels.append(__num_to_binary_list(image[y,x,c], bit_range))
      value_lst = __linearize_list(binary_channels)
      image_out[y,x] = int(__binary_list_to_num(value_lst))
  return image_out

def to_multi_channel(image, channel_num, bit_range=8):
  image_out = np.empty(image.shape + (channel_num,), dtype=np.int32)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      binary_lst = __num_to_binary_list(image[y,x], channel_num * bit_range)
      binary_channels = __delinearize_list(binary_lst, channel_num)
      for c in range(channel_num):
        image_out[y,x,c] = __binary_list_to_num(binary_channels[c])
  return image_out

def mse(image_a, image_b):
  return np.square(image_a - image_b).mean(axis=None)

def psnr(image_a, image_b, value_range):
  return 10 * math.log10((value_range - 1)**2/mse(image_a, image_b))

# def prepare_plot(testimage_len):
#   figsize = 4
#   fig = plt.figure(figsize=(figsize * figsize, testimage_len))
#   plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

# def plot_im(rows, columns, y, x, im, title):
#     plt.subplot(rows, columns, y * columns + x + 1); 
#     plt.imshow(im, interpolation="bicubic", cmap="gray"); 
#     plt.xticks([]); plt.yticks([])
#     plt.title(title)