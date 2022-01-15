import numpy as np
import torch as tc
import higra as hg

def loss_ranked_selection(saliency_measure, importance_measure, num_positives, margin, p=1):
  sorted_indices = tc.argsort(importance_measure, descending=True)
  saliency_measure = saliency_measure[sorted_indices]

  if len(saliency_measure) <= num_positives:
    return tc.sum(tc.relu(margin - saliency_measure)**p)
  else:
    return tc.sum(tc.relu(margin - saliency_measure[:num_positives])**p) + tc.sum(saliency_measure[num_positives:]**p)


def attribute_depth(tree, altitudes):
  return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)


def attribute_saddle_nodes(tree, attribute):
  max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
  child_index = hg.attribute_child_number(tree)
  main_branch = child_index == max_child_index[tree.parents()]
  main_branch[:tree.num_leaves()] = True

  saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices())[tree.parents()], main_branch)
  base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices()), main_branch)
  return saddle_nodes, base_nodes


def loss_maxima(comp_tree, graph, image, saliency_measure, importance_measure, num_target_maxima, margin=1, p=1):
  if not saliency_measure in ["altitude", "dynamics"]:
    raise ValueError("Saliency_measure can be either 'altitude' or 'dynamics'")

  if not importance_measure in ["altitude", "dynamics", "area", "volume"]:
    raise ValueError("Importance_measure can be either 'altitude', 'dynamics', 'area', or 'volume'")
  
  tree, altitudes = comp_tree(graph, image)
  altitudes_np = altitudes.detach().numpy()

  extrema = hg.attribute_extrema(tree, altitudes_np)
  extrema_indices = np.arange(tree.num_vertices())[extrema]
  extrema_altitudes = altitudes[tc.from_numpy(extrema_indices).long()] # added .long()

  if importance_measure == "dynamics" or saliency_measure == "dynamics":
    depth = attribute_depth(tree, altitudes_np)
    saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
    extrema_dynamics = extrema_altitudes - altitudes[saddle_nodes[extrema_indices].long()] # added .long()

  if importance_measure == "area":
    area = hg.attribute_area(tree)
    pass_nodes, base_nodes = attribute_saddle_nodes(tree, area)
    extrema_area = tc.from_numpy(area[base_nodes[extrema_indices]])

  if importance_measure == "volume":
    volume = hg.attribute_volume(tree, altitudes_np)
    pass_nodes, base_nodes = attribute_saddle_nodes(tree, volume)
    extrema_volume = tc.from_numpy(volume[base_nodes[extrema_indices]])
  
  if saliency_measure == "altitude":
    saliency = extrema_altitudes
  elif saliency_measure == "dynamics":
    saliency = extrema_dynamics

  if importance_measure == "altitude":
    importance = extrema_altitudes
  elif importance_measure == "dynamics":
    importance = extrema_dynamics
  elif importance_measure == "area":
    importance = extrema_area
  elif importance_measure == "volume":
    importance = extrema_volume

  return loss_ranked_selection(saliency, importance, num_target_maxima, margin, p)


def loss_l2(image, observation):
  return tc.mean((image - observation)**2)


def loss_tv(image, p=2):
  return tc.mean(tc.abs(image[:,1:] - image[:,:-1])**p) + tc.mean(tc.abs(image[1:,:] - image[:-1,:])**p)


def loss(max_tree, graph, im_tc, image):
  l = (loss_l2(image, im_tc) + 
       0.05 * loss_maxima(max_tree, graph, image, "dynamics", "dynamics", num_target_maxima=1, margin=0.5) + 
       0.5 * loss_tv(image, 2))
  return l

def loss2(max_tree, graph, im_tc, image):
  l = (loss_l2(image, im_tc) + 
       0.05 * loss_maxima(max_tree, graph, image, "altitude", "altitude", num_target_maxima=1, margin=0.5) + 
       0.5 * loss_tv(image, 2))
  return l
