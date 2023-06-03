import torch
from torch.optim.optimizer import Optimizer

class Tiger(Optimizer):
  r"""A Pytorch Implementation of Tiger."""

  def __init__(self, params, lr=1e-3, beta=0.965, weight_decay=1):
    """Initialize the hyperparameters.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: 0.965)
        weight_decay (float, optional): weight decay coefficient (default: 0.01)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= beta < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(beta))
    defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
    super().__init__(params, defaults)
    
  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    Returns:
        the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        
        state = self.state[p]
        if len(state) == 0:
          state['exp_avg'] = torch.zeros_like(p)
                
        exp_avg = state['exp_avg']
        beta=group['beta']
        exp_avg.mul_(beta).add_(p.grad, alpha=1 - beta)
        
        update = torch.sign(exp_avg)
        update.mul_( (exp_avg.abs()>group['lr']/10).type_as(update) )
        update.add_(p.data,alpha=group['weight_decay'])
        
        p.add_(update,alpha=-group['lr'])

    return loss
