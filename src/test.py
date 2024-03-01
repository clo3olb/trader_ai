import torch



# check whether I can use GPU

print(torch.cuda.is_available())

print(torch.cuda.current_device())