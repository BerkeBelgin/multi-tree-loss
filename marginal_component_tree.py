import numpy as np
import torch as tc
import higra as hg

from torch.autograd import Function
from torch.nn import Module

class ComponentTree(Module):
    def __init__(self, tree_type):
        super().__init__()
        tree_types = ("max", "min", "tos")
        if tree_type not in tree_types:
          raise ValueError("Unknown tree type " + str(tree_type) + " possible values are " + " ".join(tree_types))

        self.tree_type = tree_type

    def forward(self, graph_lst, vertex_weights):
        altitudes_merged = ComponentTreeFunction.apply(graph_lst, vertex_weights, self.tree_type)
        return altitudes_merged.tree_lst, altitudes_merged



class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, tree_type="max", plateau_derivative="full"):
    tree_lst = []
    altitudes_merged = np.array([])
    if tree_type == "max":
      for i in range(vertex_weights.size(dim=2)):
        tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy()[:,:,i])
        tree_lst.append(tree)
        altitudes_merged = np.append(altitudes_merged, altitudes)
    elif tree_type == "min":
      for i in range(vertex_weights.size(dim=2)):
        tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy()[:,:,i])
        tree_lst.append(tree)
        altitudes_merged = np.append(altitudes_merged, altitudes)
    elif tree_type == "tos":
      for i in range(vertex_weights.size(dim=2)):
        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy()[:,:,i])
        tree_lst.append(tree)
        altitudes_merged = np.append(altitudes_merged, altitudes)
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    
    ctx.saved = (tree_lst, graph, plateau_derivative)
    altitudes_merged = tc.from_numpy(altitudes_merged).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes_merged.tree_lst = tree_lst
    return altitudes_merged

  @staticmethod
  def backward(ctx, grad_output):
    tree_lst, graph, plateau_derivative = ctx.saved
    im_shape = graph.shape + (len(tree_lst),)
    image = np.empty(im_shape)
    # grad_output_lst = np.array_split(grad_output, im_shape[2])
    grad_output_lst = []
    
    j = 0
    for i in range(im_shape[2]):
      grad_len = len(tree_lst[i].parents())
      grad_output_lst.append(grad_output[j:j+grad_len])
      j+= grad_len
    if plateau_derivative:
      for i in range(im_shape[2]):
        grad_in = grad_output_lst[i][tree_lst[i].parents()[:tree_lst[i].num_leaves()]]
        image[:,:,i] = hg.delinearize_vertex_weights(grad_in, graph)
    else: # It will be implemented
      for i in range(im_shape[2]):
        leaf_parents = tree_lst[i].parents()[:tree_lst[i].num_leaves()]
        _, indices = np.unique(leaf_parents, return_index=True)
        grad_in = tc.zeros((tree_lst[i].num_leaves(),), dtype=grad_output[i].dtype)
        grad_in[indices] = grad_output[i][leaf_parents[indices]]
        image[:,:,i] = hg.delinearize_vertex_weights(grad_in, graph)
    return None, tc.from_numpy(image), None
