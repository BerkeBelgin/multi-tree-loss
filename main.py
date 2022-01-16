import numpy as np
import torch as tc

import image_handler as ih

import marginal_optimizer as mo
import marginal_loss_functions as mlf

import vectorial_optimizer as vo
import vectorial_loss_functions as vlf

import time

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

def train_model(loss_type, image, sal_mes, imp_mes, num_maxima=1, tree_type="max", lr=0.001, itr=1000, min_lr=1e-6, debug=False):
  if(loss_type == "marg"):
    loss_func = lambda comp_tree, graph, img: mlf.loss_maxima(comp_tree, graph, img, sal_mes, imp_mes, num_maxima)
    opt = mo.Optimizer(tree_type, loss_func, lr)
  elif(loss_type == "vec"):
    loss_func = lambda comp_tree, graph, img: vlf.loss_maxima(comp_tree, graph, img, sal_mes, imp_mes, num_maxima)
    opt = vo.Optimizer(tree_type, loss_func, lr)
  else:
    raise ValueError("Loss type can be either 'marg' or 'vec'")
  opt.fit(tc.from_numpy(image.copy()), itr, debug, min_lr)
  return opt.best.detach().numpy()


input("Welcome to Maxima Loss Demo.\nPress Enter to Start...")


# image
img_path = "https://media.gettyimages.com/photos/rolling-tuscany-landscape-picture-id537856787?s=612x612"
image = prepare_image(img_path, y_e=200, x_e=300, dims=[0,1,2])
ih.show_image(image)
# image


input("This is the test image.\nPress Enter to Proceed...")


# noise
image_noisy = ih.sp_noise(image, 0.06, wb=True)
ih.show_image(image_noisy)
# noise


input("This is the image with salt & pepper noise with 6% noise probability.\nPress Enter to Proceed...")


# test image unit
start_time = time.time()
image_reduced = train_model("marg", image_noisy, "altitude", "altitude", 0, itr=400, min_lr=None, debug=False)
# ih.show_image(image_reduced)

image_reduced = ih.invert_image(image_reduced)
image_reduced = train_model("marg", image_reduced, "altitude", "altitude", 0, itr=500, min_lr=None, debug=False)
image_reduced = ih.invert_image(image_reduced)
end_time = time.time()
ih.show_image(image_reduced)

print("\nNoise is reduced by marginal processing")
print("Execution Time: {:.2f}sec".format(end_time - start_time))
print("Learning Rate = 0.001")
print("Iterations = 900")
print("Original - Noised  Image MSE: " + str(ih.mse(image, image_noisy)))
print("Original - Reduced Image MSE: " + str(ih.mse(image, image_reduced)))
# test image unit


input("Press Enter to proceed...")


# test image unit
start_time = time.time()
image_sc = convert_to_single_channel(image_noisy, 3, 8)
# ih.show_image(image_sc)

image_reduced_sc = train_model("vec", image_sc, "altitude", "altitude", itr=200, min_lr=None, debug=False)
# ih.show_image(image_reduced_sc)

image_reduced_sc = ih.invert_image(image_reduced_sc)
image_reduced_sc = train_model("vec", image_reduced_sc, "altitude", "altitude", itr=200, min_lr=None, debug=False)
image_reduced_sc = ih.invert_image(image_reduced_sc)
# ih.show_image(image_reduced_sc)

image_reduced = convert_to_multi_channel(image_reduced_sc, 3, 8)
end_time = time.time()
ih.show_image(image_reduced)

print("\nNoise is reduced by vectorial processing")
print("Execution Time: {:.2f}sec".format(end_time - start_time))
print("Learning Rate = 0.001")
print("Iterations = 400")
print("Original - Noised  MSE: " + str(ih.mse(image, image_noisy)))
print("Original - Reduced MSE: " + str(ih.mse(image, image_reduced)))
# test image unit


input("Press Enter to proceed...")


# noise
image_noisy = ih.sp_noise(image, 0.06, wb=False)
ih.show_image(image_noisy)
# noise


input("This time, salt & pepper noise is applied to every channel of the image individually. This causes noises on different points in different channels.\nPress Enter to proceed...")


# test image unit
start_time = time.time()
image_reduced = train_model("marg", image_noisy, "altitude", "altitude", 0, itr=400, min_lr=None, debug=False)
# ih.show_image(image_reduced)

image_reduced = ih.invert_image(image_reduced)
image_reduced = train_model("marg", image_reduced, "altitude", "altitude", 0, itr=500, min_lr=None, debug=False)
image_reduced = ih.invert_image(image_reduced)
end_time = time.time()
ih.show_image(image_reduced)

print("\nNoise is reduced by marginal processing")
print("Execution Time: {:.2f}sec".format(end_time - start_time))
print("Learning Rate = 0.001")
print("Iterations = 900")
print("Original - Noised  MSE: " + str(ih.mse(image, image_noisy)))
print("Original - Reduced MSE: " + str(ih.mse(image, image_reduced)))
# test image unit


input("Press Enter to proceed...")


# test image unit
start_time = time.time()
image_sc = convert_to_single_channel(image_noisy, 3, 8)
# ih.show_image(image_sc)

image_reduced_sc = train_model("vec", image_sc, "altitude", "altitude", itr=200, min_lr=None, debug=False)
# ih.show_image(image_reduced_sc)

image_reduced_sc = ih.invert_image(image_reduced_sc)
image_reduced_sc = train_model("vec", image_reduced_sc, "altitude", "altitude", itr=200, min_lr=None, debug=False)
image_reduced_sc = ih.invert_image(image_reduced_sc)
# ih.show_image(image_reduced_sc)

image_reduced = convert_to_multi_channel(image_reduced_sc, 3, 8)
end_time = time.time()
ih.show_image(image_reduced)

print("\nNoise is reduced by vectorial processing")
print("Execution Time: {:.2f}sec".format(end_time - start_time))
print("Learning Rate = 0.001")
print("Iterations = 400")
print("Original - Noised  MSE: " + str(ih.mse(image, image_noisy)))
print("Original - Reduced MSE: " + str(ih.mse(image, image_reduced)))
# test image unit
