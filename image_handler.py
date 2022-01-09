import numpy as np
import matplotlib.pyplot as plt
import imageio

def draw_Gaussian_2d(im, x0, y0, sigma_X, sigma_Y, theta, A):
  X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))

  a = np.cos(theta)**2/(2*sigma_X**2) + np.sin(theta)**2/(2*sigma_Y**2)
  b = -np.sin(2*theta)/(4*sigma_X**2) + np.sin(2*theta)/(4*sigma_Y**2)
  c = np.sin(theta)**2/(2*sigma_X**2) + np.cos(theta)**2/(2*sigma_Y**2)

  Z = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
  
  im += Z

def gen_multi_max(noise):
  im = np.zeros((100, 100))
  draw_Gaussian_2d(im, 30, 40, 18, 25, -0.3, 1.7)
  draw_Gaussian_2d(im, 20, 90, 5, 5, 0, 2)
  draw_Gaussian_2d(im, 80, 30, 12, 5, 1.8, 1.9)
  draw_Gaussian_2d(im, 80, 80, 15, 15, 0, 1.6)

  im = im / im.max()
  if noise > 0:
    im = im + np.random.randn(*im.shape) * noise
    im = im / im.max()
  return im



def normalize_image(image):
    return image / image.max()

def invert_image(image):
    return 1 - image

def noise_image(image, strength):
    return image + np.random.rand(*image.shape) * strength

def get_image(path):
    image = imageio.imread(path)
    image = normalize_image(image)
    
    if(image.ndim == 2):
        return [image]
    elif(image.ndim == 3):
        return image
    else:
        raise ValueError("The image must be 2 or 3 dimensional.")

def show_image(image):
    if(len(image) == 3):
        plt.imshow(image)
    for i in range(len(image)):
        plt.imshow(image[i])

# def prepare_plot(testimage_len):
#   figsize = 4
#   fig = plt.figure(figsize=(figsize * figsize, testimage_len))
#   plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

# def plot_im(rows, columns, y, x, im, title):
#     plt.subplot(rows, columns, y * columns + x + 1); 
#     plt.imshow(im, interpolation="bicubic", cmap="gray"); 
#     plt.xticks([]); plt.yticks([])
#     plt.title(title)