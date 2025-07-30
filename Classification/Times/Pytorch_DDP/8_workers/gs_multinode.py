import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from NFNet import NFNet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time
from torch.distributed.elastic.multiprocessing.errors import record


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

def load_data(x_train, y_train):
    x_train = torch.load(x_train)
    y_train = torch.load(y_train)
    return x_train, y_train

# Función de entrenamiento con rank explícito
def train():
    
    dist.init_process_group(backend="nccl")
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = dist.get_rank()
    
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    model = NFNet().to(device_id)
    model = model.apply(init_weights_kaiming)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    
    x_train, y_train = load_data("/PATH_TO_DATASET",
            "/PATH_TO_DATASET")
    
    dataset = MyTensorDataset(x_train, y_train)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.000001, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0000001, T_max=10)
    for epoch in range(50):
        sampler.set_epoch(epoch)
        if epoch == 1:
            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=10)    
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device_id)
            y = y.to(device_id)
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

    end_time = time.time()
    
    if rank == 0:
        print("Execution Time: " + str(end_time - start_time), flush=True)
        torch.save(model.state_dict(), "modelo.pth")

    dist.barrier()
    dist.destroy_process_group()


# Main que lanza 16 procesos y pasa rank explícitamente
@record
def main():
    train()

if __name__ == "__main__":
    main()
