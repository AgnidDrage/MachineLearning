import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.linear_stack(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)

#x_train = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]], dtype=torch.float).to(device)
x_train = torch.arange(0,110,10).float().to(device)
x_train = x_train.view(-1, 1) # Reshape to (10, 1) because Linear layer expects 2D input
y_train = torch.add(x_train, 273.15).float().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(10000):
    predict_y = model(x_train)
    loss = criterion(predict_y, y_train)
    if loss < 0.001:
        print(f"Epoch: {epoch} Loss: {loss.item()}")
        break
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {loss.item()}")

y_pred = model(torch.tensor([[10]], dtype=torch.float).to(device))
print(y_pred.item())