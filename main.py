import numpy as np
import torch as tc
import higra as hg

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
# img_path = "https://media.gettyimages.com/photos/rolling-tuscany-landscape-picture-id537856787?s=612x612"
# image = prepare_image(img_path, y_e=200, x_e=300, dims=[0,1,2])
# ih.show_image(image)

# image_noisy = ih.sp_noise(image, 0.06, wb=False)
# ih.show_image(image_noisy)

# image_sc = convert_to_single_channel(image_noisy, 3, 8)
# ih.show_image(image_sc)

# loss_func = lambda comp_tree, graph, img: vlf.loss_maxima(comp_tree, graph, img, "altitude", "altitude", 1)
# opt = vo.Optimizer("max", loss_func, lr=0.001)
# opt.fit(tc.from_numpy(image_sc.copy()), iter=200, min_lr=None, debug=True)
# image_reduced_sc = opt.best.detach().numpy()
# ih.show_image(image_reduced_sc)

# image_reduced_sc = ih.invert_image(image_reduced_sc)
# loss_func = lambda comp_tree, graph, img: vlf.loss_maxima(comp_tree, graph, img, "altitude", "altitude", 1)
# opt = vo.Optimizer("max", loss_func, lr=0.001)
# opt.fit(tc.from_numpy(image_reduced_sc.copy()), iter=200, min_lr=None, debug=True)
# image_reduced_sc = opt.best.detach().numpy()
# image_reduced_sc = ih.invert_image(image_reduced_sc)
# ih.show_image(image_reduced_sc)

# image_reduced = convert_to_multi_channel(image_reduced_sc, 3, 8)
# ih.show_image(image_reduced)

# print("Noised MSE: " + str(ih.mse(image, image_noisy)) + ", Reduced MSE: " + str(ih.mse(image, image_reduced)))
# # test image unit





# test image unit
img_path = "https://media.gettyimages.com/photos/rolling-tuscany-landscape-picture-id537856787?s=612x612"
image = prepare_image(img_path, y_e=200, x_e=300, dims=[0,1,2])
# ih.show_image(image)

image_noisy = ih.sp_noise(image, 0.06, wb=False)
# ih.show_image(image_noisy)

loss_func = lambda comp_tree, graph, img: mlf.loss_maxima(comp_tree, graph, img, "altitude", "altitude", 0)
opt = mo.Optimizer("max", loss_func, lr=0.001)
opt.fit(tc.from_numpy(image_noisy.copy()), iter=400, min_lr=None, debug=True)
image_reduced = opt.best.detach().numpy()
ih.show_image(image_reduced)

image_reduced = ih.invert_image(image_reduced)
loss_func = lambda comp_tree, graph, img: mlf.loss_maxima(comp_tree, graph, img, "altitude", "altitude", 0)
opt = mo.Optimizer("max", loss_func, lr=0.001)
opt.fit(tc.from_numpy(image_reduced.copy()), iter=500, min_lr=None, debug=True)
image_reduced = opt.best.detach().numpy()
image_reduced = ih.invert_image(image_reduced)
ih.show_image(image_reduced)

print("Noised MSE: " + str(ih.mse(image, image_noisy)) + ", Reduced MSE: " + str(ih.mse(image, image_reduced)))
# test image unit
