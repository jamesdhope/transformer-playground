from torch import nn, Tensor
import torch
import torch.nn.functional as F

output = ([[-1.3346,  2.1291,  4.2817,  ..., -1.2973,  0.2866,  1.4358]])
token_probs = F.softmax(output[-1, 0, :] / 0.7, dim=-1)

print(token_probs)