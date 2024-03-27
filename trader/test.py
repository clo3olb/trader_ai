import torch

t = torch.tensor([[10, 2], [3, 4]])
print(t)
indices = torch.tensor([[1, 1], [1, 1]])

print(indices)
result = t.gather(1, indices)
print(result)
