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


def loss_maxima(comp_tree, graph, image, saliency_measures, importance_measures, num_target_maximas, margin=1, p=1):
  img_shape = image.shape
  
  if isinstance(saliency_measures, str):
    saliency_measures = [saliency_measures] * img_shape[2]
  if isinstance(importance_measures, str):
    importance_measures = [importance_measures] * img_shape[2]
  if isinstance(num_target_maximas, int):
    num_target_maximas = [num_target_maximas] * img_shape[2]
    
  if not set(saliency_measures).issubset(["altitude", "dynamics"]):
    raise ValueError("Saliency_measure can be either 'altitude' or 'dynamics'")

  if not set(importance_measures).issubset(["altitude", "dynamics", "area", "volume"]):
    raise ValueError("Importance_measure can be either 'altitude', 'dynamics', 'area', or 'volume'")
  
  tree_lst, altitudes_merged = comp_tree(graph, image)
  total_loss = tc.tensor(0.0, requires_grad=True)
  
  j = 0
  for i in range(img_shape[2]):
    tree = tree_lst[i]
    altitudes_len = len(tree.parents())
    altitudes = altitudes_merged[j:j+altitudes_len]
    altitudes_np = altitudes.detach().numpy()
    
    j+= altitudes_len
    
    extrema = hg.attribute_extrema(tree, altitudes_np)
    extrema_indices = np.arange(tree.num_vertices())[extrema]
    extrema_altitudes = altitudes[tc.from_numpy(extrema_indices).long()] # added .long()
  
    if importance_measures[i] == "dynamics" or saliency_measures[i] == "dynamics":
      depth = attribute_depth(tree, altitudes_np)
      saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
      extrema_dynamics = extrema_altitudes - altitudes[saddle_nodes[extrema_indices].long()] # added .long()
  
    if importance_measures[i] == "area":
      area = hg.attribute_area(tree)
      pass_nodes, base_nodes = attribute_saddle_nodes(tree, area)
      extrema_area = tc.from_numpy(area[base_nodes[extrema_indices]])
  
    if importance_measures[i] == "volume":
      volume = hg.attribute_volume(tree, altitudes_np)
      pass_nodes, base_nodes = attribute_saddle_nodes(tree, volume)
      extrema_volume = tc.from_numpy(volume[base_nodes[extrema_indices]])
    
    if saliency_measures[i] == "altitude":
      saliency = extrema_altitudes
    elif saliency_measures[i] == "dynamics":
      saliency = extrema_dynamics
  
    if importance_measures[i] == "altitude":
      importance = extrema_altitudes
    elif importance_measures[i] == "dynamics":
      importance = extrema_dynamics
    elif importance_measures[i] == "area":
      importance = extrema_area
    elif importance_measures[i] == "volume":
      importance = extrema_volume
    
    total_loss = total_loss + loss_ranked_selection(saliency, importance, num_target_maximas[i], margin, p)
  return total_loss


def loss_l2(image, observation):
  return tc.mean((image - observation)**2)


def loss_tv(image, p=2):
  return tc.mean(tc.abs(image[:,1:,:] - image[:,:-1,:])**p) + tc.mean(tc.abs(image[1:,:,:] - image[:-1,:,:])**p)


def loss(max_tree, graph, im_tc, image):
  l = (loss_l2(image, im_tc) + 
       0.05 * loss_maxima(max_tree, graph, image, "dynamics", "dynamics", num_target_maximas=1, margin=0.5) + 
       0.5 * loss_tv(image, 2))
  return l

def loss2(max_tree, graph, im_tc, image):
  l = (loss_l2(image, im_tc) + 
       0.05 * loss_maxima(max_tree, graph, image, "altitude", "altitude", num_target_maximas=1, margin=0.5) + 
       0.5 * loss_tv(image, 2))
  return l
