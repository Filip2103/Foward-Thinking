import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Batch_size = 64 omogucava mini batch treniranje
def get_dataloaders(batch_size=64):
    # Funkcije koje menjaju reprezentaciju ulaza
    transform = transforms.Compose([
        # PIL slika -> torch tensor i sklairanje piksela
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #Ucitavanje traning skupa
    full_train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    #Ucitavanje test skupa
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    """
    Delimo skupove podataka na trening i validacioni skup.
    Delimo 60k -> 48k trening + 12k val.
    Forward thinking koristi validacioni skup da odluci da li novi sloj pobljsava model
    """
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # SGD zahteva random raspored i smanjuje bias u batch-u (shuffle True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Evaluacija treba da bude deterministicka i redlosed je nebitan (shuffle False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader 
