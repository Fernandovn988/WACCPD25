import torch
from NFNet import NFNet
from torch.utils.data import DataLoader


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

def main():
    # Cargar modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NFNet().to(device)
    model.load_state_dict(torch.load("./modelo.pth"))
    model.eval()
    x_test, y_test = load_data("/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/test/x_test_64.pt",
            "/gpfs/scratch/bsc19/bsc019756/Neural_Network_With_GLAI/Classification/Dataset/test/y_test.pt")
    test_dataset = MyTensorDataset(x_test, y_test)
    # Data de evaluación
    test_loader = DataLoader(test_dataset, batch_size=64)
    # Evaluación
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    accuracy = correct / total
    print(f"[Eval] Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
