import torch
from torch import nn
import matplotlib.pyplot as plt

# Create data tensors
a = torch.tensor([0, 0, 1, 1], dtype=torch.float)
b = torch.tensor([0, 1, 0, 1], dtype=torch.float)
spectedOutput = torch.tensor([[0, 1, 1, 0]], dtype=torch.float)

# Create the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.linear(x)))

model = Model()

# Create the loss function
criterion = nn.L1Loss()

# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Store the losses
train_loss_values = []
epoch_count = []

# Train the model
epochs = 1000

print(model.state_dict())
breakpoint()
for epoch in range(epochs):
    # Forward pass
    y_pred = model(torch.stack((a, b), dim=1))
    loss = criterion(y_pred, spectedOutput.T)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print out what is happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())

# Plot the loss
plt.plot(epoch_count, train_loss_values, 'r--')
plt.show()

# predict
model.eval()
print("Predictions:")
print(model(torch.stack((a, b), dim=1)))
print("Spected output:")
print(spectedOutput.T)