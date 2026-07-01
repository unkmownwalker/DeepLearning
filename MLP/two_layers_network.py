import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

#gpu setup
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#model definition
D_i,D_k,D_o=10,40,5
model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

#parameters initialization
def weight_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(weight_init)

#move to gpu
model.to(device)

#settings
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),
                           lr=0.001,betas=(0.9,0.999),eps=10e-8)

#data(cpu)
x = torch.randn(100, D_i)
y = torch.randn(100, D_o)
data_loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=True)

#train
for epoch in range(100):
    epoch_loss=0.0
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch%10==0:
        print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
