import numpy as np
import torch as tc
import higra as hg

from optimizer import Optimizer
from component_tree import ComponentTree
from loss_functions import loss_maxima, loss, loss2
import image_handler as ih

np.random.seed(2022)
max_tree = ComponentTree("max")

image = ih.get_image("https://i.stack.imgur.com/B5EhD.png")
# image = ih.invert_image(image)

graphs = []
for i in range(len(image)):
    graphs.append(hg.get_8_adjacency_implicit_graph(image[i].shape))

image = ih.noise_image(image, 0.01)
torch_image = tc.from_numpy(image.copy())

print("Input image")
ih.show_image(image)

loss_func = lambda img: loss2(max_tree, graphs, torch_image, img)
# loss_func = lambda img: loss_maxima(max_tree, graph, img, "dynamics", "dynamics", num_target_maxima=2, margin=1, p=1)

opt = Optimizer(loss_func, lr=0.001)
opt.fit(torch_image, iter=4000, debug=True)

print("Filtered image")
ih.show_image(opt.best.detach().numpy())

image = ih.invert_image(image)
ih.show_image(image)
