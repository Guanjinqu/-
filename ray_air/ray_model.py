import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ray.train as train
from ray.air import session
# Download training data from open datasets.
# Download training data from open datasets.
training_data = datasets.MNIST(
    root='./data',
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=False,
    transform=ToTensor(),
)



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.ln1 = nn.Linear(784,500)
        self.ln2 = nn.Linear(500,10)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        return self.ln2(self.relu(self.ln1(x)))

#model = NeuralNetwork().to(device)

#loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // session.get_world_size()  # Divide by word size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # We don't need this anymore! Ray Train does this automatically:
        # X, y = X.to(device), y.to(device)  

        # Compute prediction error
        X = X.reshape(-1,28*28)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // session.get_world_size()  # Divide by word size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1,28*28)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_func(config: dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    batch_size_per_worker = batch_size // session.get_world_size()
    

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size_per_worker)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_per_worker)
    
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for t in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_epoch(test_dataloader, model, loss_fn)
        session.report(dict(loss=test_loss))

    print("Done!")

from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig


trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
    scaling_config=ScalingConfig(num_workers=5, use_gpu=False),
)
result = trainer.fit()
print(f"Last result: {result.metrics}")