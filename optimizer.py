import torch as tc
# import matplotlib.pyplot as plt

class Optimizer:
  def __init__(self, loss, lr, optimizer="adam"):
    self.loss_function = loss
    self.history = []
    self.optimizer = optimizer
    self.lr = lr
    self.best = None
    self.best_loss = float("inf")

  def fit(self, data, iter=1000, debug=False, min_lr=1e-6):
    data = data.clone().requires_grad_(True)
    
    if self.optimizer == "adam":
      optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
    else:
      optimizer = tc.optim.SGD([data], lr=self.lr)

    if min_lr:
      lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100)

    for t in range(iter):
      optimizer.zero_grad()
      loss = self.loss_function(tc.relu(data))
      loss.backward()
      optimizer.step()  
      loss_value = loss.item()
      
      self.history.append(loss_value) 
      if loss_value < self.best_loss:
        self.best_loss = loss_value
        self.best = tc.relu(data).clone()
        
      if min_lr:
        lr_scheduler.step(loss_value)
        if optimizer.param_groups[0]['lr'] <= min_lr:
          break

      if debug and t % debug == 0:
        print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
    return self.best

  # def show_history(self):
    # plt.plot(self.history)
    # plt.show()
