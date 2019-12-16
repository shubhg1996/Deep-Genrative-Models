import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid
    x_fake = g(z)
    logits_real = d(x_real)
    logits_fake = d(x_fake)
    d_loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y_real, *, device):
    '''
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    - g_loss (torch.Tensor): nonsaturating conditional generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR CODE STARTS HERE
    x_fake = g(z, y_real)
    logits_real = d(x_real, y_real)
    logits_fake = d(x_fake, y_fake)
    d_loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    d_loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    - g_loss (torch.Tensor): wasserstein generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    x_fake = g(z)
    d_loss_real = -torch.mean(d(x_real))
    d_loss_fake = torch.mean(d(x_fake))
    g_loss = -d_loss_fake
    alpha = torch.rand(batch_size, device=device)
    alpha = alpha.view(-1,1,1,1)
    x_alpha = alpha*x_fake + (1-alpha)*x_real
    # x_alpha = torch.autograd.Variable(x_alpha, requires_grad=True)
    d_loss_alpha = d(x_alpha)
    grads = torch.autograd.grad(outputs=d_loss_alpha, inputs=x_alpha, grad_outputs=torch.ones_like(d_loss_alpha), create_graph=True)
    grad_loss = torch.mean((grads[0].view(batch_size, -1).norm(2, dim=1) - 1)**2)
    d_loss = d_loss_fake + d_loss_real + 10*grad_loss
    # YOUR CODE ENDS HERE

    return d_loss, g_loss
