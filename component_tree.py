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

    def forward(self, graph, vertex_weights):
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, self.tree_type)
        return altitudes.tree, altitudes



class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, tree_type="max", plateau_derivative="full"):
    if tree_type == "max":
      tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "min":
      tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "tos":
      tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy())
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    ctx.saved = (tree, graph, plateau_derivative)
    altitudes = tc.from_numpy(altitudes).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes.tree = tree
    return altitudes

  @staticmethod
  def backward(ctx, grad_output):
    tree, graph, plateau_derivative = ctx.saved
    if plateau_derivative:
      grad_in = grad_output[tree.parents()[:tree.num_leaves()]]
    else:
      leaf_parents = tree.parents()[:tree.num_leaves()]
      _, indices = np.unique(leaf_parents, return_index=True)
      grad_in = tc.zeros((tree.num_leaves(),), dtype=grad_output.dtype)
      grad_in[indices] = grad_output[leaf_parents[indices]]
    return None, hg.delinearize_vertex_weights(grad_in, graph), None
