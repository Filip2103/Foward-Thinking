import torch
import torch.nn as nn
from src.model import LayerwiseMLP
from src.data import get_dataloaders
from src.train import train_model

"""
Ovaj fajl implementira sledeci algoritam:

Repeat:
    Zamrzni stare slojeve
    Dodaj novi sloj
    Treniraj plitku mrežu
    Ako postoji poboljšanje → zadrži sloj
    U suprotnom → zaustavi postupak
Evauliraj finalni model

"""
def freeze_for_new_layer_training(model):
    # 1. zamrzni sve parametre
    for param in model.parameters():
        param.requires_grad = False

    # 2. odmrzni POSLEDNJI skriveni Linear sloj (novi sloj)
    hidden_linears = [m for m in model.hidden_layers if isinstance(m, nn.Linear)]
    if len(hidden_linears) > 0:
        # Odmrzni poslednji skriveni Linear sloj 
        for p in hidden_linears[-1].parameters():
            p.requires_grad = True

    # 3. odmrzni output sloj
    for p in model.output_layer.parameters():
        p.requires_grad = True


def layerwise_training(
    input_dim=784,
    hidden_dim=128,
    output_dim=10,
    max_layers=3,
    device="cpu"
):
    # Ucitavanje dataseta
    train_loader, val_loader, test_loader = get_dataloaders()

    print("=== Training base model ===")

    # Bazni model 
    # In -> Hidden1 -> Output
    hidden_dims = [hidden_dim]
    model = LayerwiseMLP(input_dim, hidden_dims, output_dim)

    # Treniranje baznog modela
    best_state, best_val_acc = train_model(
        model, train_loader, val_loader, device=device
    )

    # Cuvanje konfiguracije (pamti koliko slojeva ima najbolji model)
    best_hidden_dims = hidden_dims.copy()
    print(f"Base model val acc: {best_val_acc:.4f}")

    # Petlja za dodavanje slojeva
    for layer_idx in range(2, max_layers + 1):
        print(f"\n=== Adding layer {layer_idx} ===")

        # Dodavanje sloja, sada mreza ima Hidden1 -> Hidden2 -> Output
        hidden_dims.append(hidden_dim)
        new_model = LayerwiseMLP(input_dim, hidden_dims, output_dim)

        # Ucitavanje starih tezina. TO znaci da Hidden1 dobija stare tezine, a Hidden2 ostaje random
        new_model.load_state_dict(best_state, strict=False)

        # Reset output sloja. Bez ovoga bi output bio prilagodjen starim feature-ima
        new_model.output_layer.reset_parameters()

        # Zamrzavanje starih slojeva, treniranje poslednjeg skrivenog i output sloja
        freeze_for_new_layer_training(new_model)

        # Treniranje nove mreze, ali po novim parametrima
        state, val_acc = train_model(
            new_model, train_loader, val_loader, device=device
        )

        print(f"Val acc with {layer_idx} layers: {val_acc:.4f}")

        # Evaluacija poboljsanja
        if val_acc > best_val_acc:
            print("Improvement detected. Keeping new layer.")
            best_val_acc = val_acc
            best_state = state
            best_hidden_dims = hidden_dims.copy()
        else:
            print("No improvement. Stopping layer-wise training.")
            break

    print("\n=== Final evaluation on test set ===")

    # Finalna evaluacija
    final_model = LayerwiseMLP(
        input_dim, best_hidden_dims, output_dim
    )
    final_model.load_state_dict(best_state)
    final_model.to(device)
    final_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        # Test petlja
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            outputs = final_model(x)
            _, preds = torch.max(outputs, 1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}")

    return best_hidden_dims, best_val_acc, test_acc, best_state
