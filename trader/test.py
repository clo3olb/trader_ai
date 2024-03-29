import matplotlib.pyplot as plt
import torch

learning_rate = 0.01
num_iterations = 1000

initial_x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
initial_y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

# Reshape the input tensor
x = initial_x.view(-1, 1)
y = initial_y.view(-1, 1)

# Define the weights and bias
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)


# Define the optimizer
optimizer = torch.optim.Adam([w, b], lr=learning_rate)

# Perform gradient descentscent
for i in range(num_iterations):
    # Forward pass: compute the predicted y
    y_pred = torch.matmul(x, w) + b

    # Compute the loss
    loss = torch.mean((y_pred - y) ** 2)

    # Backward pass: compute gradients
    loss.backward()

    # Update parameters using optimizer

    optimizer.step()

    # Reset gradients
    optimizer.zero_grad()
# Print the final parameters
print("Final w:", w)
print("Final b:", b)

# plot

plt.plot(initial_x.detach().numpy(), initial_y.detach().numpy(),
         'ro', label='Original data')
plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label='Fitted line')
plt.legend()
plt.savefig('trader/plot.png')
