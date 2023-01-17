import torch
import matplotlib.pyplot as plt
from torch import nn
from model import LinearRegressionModel




# Data
X_train = torch.arange(0,10,1).float()
y_train = X_train * 2
X_test = torch.arange(10,20,1).float()
y_test = X_test * 2

# set manual seed
torch.manual_seed(42)

# Instance of the model
model_0 = LinearRegressionModel()

# Check the parameters of the model
print(model_0.state_dict())

# make predictions with the model
with torch.inference_mode():
    y_preds = model_0(X_train)

# Plot function
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})

  # Show the plot
  plt.show()

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# Plot the predictions
plot_predictions(predictions=y_preds)

# Loss function
loss_fn = nn.L1Loss() # L1Loss is the same as MAE

# Create optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01) # parameters() returns the parameters of the model and lr is the learning rate


# Train the model
torch.manual_seed(42)

# Number of epochs
epochs = 100

# Store the losses
train_loss_values = []
test_loss_values = []
epoch_count = []


for epoch in range(epochs):
    ### Training ###

    # Put model in training mode (this is the default mode)
    model_0.train()

    # 1. Forward pass on the training data using forward() method inside the model
    y_preds = model_0(X_train)

    # 2. Calculate the loss (how far off are the predictions from the true labels)
    loss = loss_fn(y_preds, y_train)

    # 3. Zero the gradients (otherwise they will accumulate)
    optimizer.zero_grad()

    # 4. Loss backward pass (calculate the gradients)
    loss.backward()

    # 5. Update the parameters (weights and biases)
    optimizer.step()

    ### Testing ###
    # Put model in evaluation mode
    model_0.eval()

    # 1. Forward pass on the test data using forward() method inside the model
    test_pred = model_0(X_test)

    # 2. Calculate the loss (how far off are the predictions from the true labels)
    test_loss = loss_fn(test_pred, y_test.type(torch.float))

    # Print out what is happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# Plot the loss
plt.plot(epoch_count, train_loss_values, 'r--')
plt.plot(epoch_count, test_loss_values, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Find our model's learned parameters
print("The model has learned the following parameters:")
print(model_0.state_dict())




# Make predictions with the model
# 1. Put model in inference mode
model_0.eval()

# 2. Setup the inference mode contex manager
with torch.inference_mode():
    # 3. Make sure the calculations are done with the model and data on the same device
    # in our case, we haven't specified a device so the model and data are on the CPU
    # model_0.to(device)
    # X_test.to(device)
    # 4. Make predictions
    y_preds = model_0(X_test)
    print(X_test)
    print(y_preds)