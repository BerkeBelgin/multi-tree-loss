import numpy as np
import torch as tc

import image_handler as ih

import marginal_optimizer as mo
import marginal_loss_functions as mlf

import vectorial_optimizer as vo
import vectorial_loss_functions as vlf

np.random.seed(42)

# for i in range(8):
#   binar = ih.num_to_binary_list(i, 8)
#   binar2 = ih.num_to_binary_list(i+1, 8)
#   binar3 = ih.num_to_binary_list(i+2, 8)
#   print([binar, binar2, binar3])
#   lin = ih.linearize_list([binar, binar2, binar3])
#   print(lin)
#   print(ih.binary_list_to_num(lin))
#   delin = ih.delinearize_list(lin, 3)
#   print(delin)
#   num = ih.binary_list_to_num(binar)
#   print(ih.binary_list_to_num(binar))
#   print(ih.binary_list_to_num(binar2))
#   print(ih.binary_list_to_num(binar3))

# img = np.array([[[1, 2, 3]]])
# img = ih.encode_channels(img, 256)
# print(img)
# img = ih.decode_channels(img, 3, 256)
# print(img)

def prepare_image(image_path, y_s=None, y_e=None, x_s=None, x_e=None, dims=0, value_range=256):
  image = ih.get_image(image_path)
  image = ih.crop_image(image, y_s, y_e, x_s, x_e)
  image = ih.get_dims(image, dims)
  return ih.normalize_image(image, value_range)

def convert_to_single_channel(image, channel_num=0, bit_range=8):
  image = ih.denormalize_image(image, 2**bit_range)
  image = ih.to_single_channel(image, bit_range)
  return ih.normalize_image(image, (2**bit_range)**channel_num)

def convert_to_multi_channel(image, channel_num=0, bit_range=8):
  image = ih.denormalize_image(image, (2**bit_range)**channel_num)
  image = ih.to_multi_channel(image, channel_num, bit_range)
  return ih.normalize_image(image, 2**bit_range)

# # test image unit
# img_path = "https://images.unsplash.com/photo-1641946732576-94e61721d705?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80"
# image = prepare_image(img_path, y_e=200, x_e=300, dims=[0,1,2])
# ih.show_image(image)

# image_noisy = ih.noise_image(image, "s&p")
# ih.show_image(image_noisy)

# image_sc = convert_to_single_channel(image_noisy, 3, 8)
# ih.show_image(image_sc)

# loss_func = lambda comp_tree, graph, img: vlf.loss_maxima(comp_tree, graph, img, "altitude", "altitude", 1)
# opt = vo.Optimizer("max", loss_func, lr=0.001)
# opt.fit(tc.from_numpy(image_sc.copy()), iter=1000, debug=True)
# image_reduced_sc = opt.best.detach().numpy()
# ih.show_image(image_reduced_sc)

# image_reduced = convert_to_multi_channel(image_reduced_sc, 3, 8)
# ih.show_image(image_reduced)

# print("Noised MSE: " + str(ih.mse(image, image_noisy)) + ", Reduced MSE: " + str(ih.mse(image, image_reduced)))
# # test image unit

# # test image unit
# img_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/A_black_image.jpg/640px-A_black_image.jpg"
# image = prepare_image(img_path, y_e=200, x_e=300, dims=0)
# ih.show_image(image)

# image_noisy = ih.draw_Gaussian_2d(image, 30, 40, 18, 25, -0.3, 1.7)
# ih.show_image(image_noisy)

# loss_func = lambda comp_tree, graph, img: mlf.loss_maxima(comp_tree, graph, img, "altitude", "volume", 1)
# opt = mo.Optimizer("max", loss_func, lr=0.001)
# opt.fit(tc.from_numpy(image_noisy.copy()), iter=1000, debug=True)
# image_reduced = opt.best.detach().numpy()
# ih.show_image(image_reduced)

# print("Noised MSE: " + str(ih.mse(image, image_noisy)) + ", Reduced MSE: " + str(ih.mse(image, image_reduced)))
# # test image unit

# test image unit
import imageio

# image = imageio.imread("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/A_black_image.jpg/640px-A_black_image.jpg")
# image = image[:200,:300,0]
# ih.show_image(image)
# image = ih.normalize_image(image, 256)


image_noisy = ih.gen_multi_max(0.05)
ih.show_image(image_noisy)

loss_func = lambda comp_tree, graph, img: vlf.loss_maxima(comp_tree, graph, img, "altitude", "volume", 2)
opt = vo.Optimizer("max", loss_func, lr=0.001)
opt.fit(tc.from_numpy(image_noisy.copy()), iter=1000, debug=True)
image_reduced = opt.best.detach().numpy()
ih.show_image(image_reduced)

# print("Noised MSE: " + str(ih.mse(image, image_noisy)) + ", Reduced MSE: " + str(ih.mse(image, image_reduced)))
# test image unit