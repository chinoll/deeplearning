import torch
##来源于:https://github.com/basiclab/GNGAN-PyTorch/blob/master/models/gradnorm.py
# Gradient norm regularization
def normalize_gradient(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    n = net_D(x, **kwargs)
    hat = []
    for f in n:
        grad = torch.autograd.grad(
            f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
        grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
        f_hat = (f / (grad_norm + torch.abs(f)))
        hat.append(f_hat)
    return hat