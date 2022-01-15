import torch as tc
import higra as hg

from marginal_component_tree import ComponentTree

class Optimizer:
  def __init__(self, tree_type, loss_func, lr, optimizer="adam"):
    self.comp_tree = ComponentTree(tree_type)
    self.loss_func = loss_func
    self.lr = lr
    self.optimizer = optimizer
    self.best = None
    self.best_loss = float("inf")
    self.history = []

  def fit(self, data, iter=1000, debug=False, min_lr=1e-6):
    data_grad = data.clone().requires_grad_(True)
    
    if self.optimizer == "adam":
      optimizer = tc.optim.Adam([data_grad], lr=self.lr, amsgrad=True)
    else:
      optimizer = tc.optim.SGD([data_grad], lr=self.lr)
    
    if min_lr:
      lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100)
    
    graph = hg.get_8_adjacency_implicit_graph(data.numpy().shape[:2])
    
    for t in range(iter):
      optimizer.zero_grad()
      loss = self.loss_func(self.comp_tree, graph, tc.relu(data_grad))
      loss.backward()
      optimizer.step()  
      loss_value = loss.item()
      
      self.history.append(loss_value) 
      if loss_value < self.best_loss:
        self.best_loss = loss_value
        self.best = tc.relu(data_grad).clone()
        
      if min_lr:
        lr_scheduler.step(loss_value)
        if optimizer.param_groups[0]['lr'] <= min_lr:
          break

      if debug and t % debug == 0:
        print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
    return self.best
