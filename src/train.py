import torch
import torch.nn as nn

"""
Genericki trening 1 mreze. U nasem slucaju koristi se za bazni model i za svaki dodatni sloj.
Ista f-ja trenira sve faze algoritma
"""
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-3,
    patience=3,
    device="cpu"
):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        # Ovim se omogucava automatsko treniranje samo novih slojeva
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:

            # Ovo ovde sluzi da pretvara x.shape[64,1,28,28] -> [64,784] zato sto nn.Linear ocekuje [batch, features] 
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            # Resetuje stare gradijente jer PyTorch sabira gradijente
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            # Racuna backprop samo za parametre sa requires_grad = True
            loss.backward()
            optimizer.step()

            # Uzima indeks najveceg logita 
            _, preds = torch.max(outputs, 1)
            # Update tacnosti, broji koliko je pogodio i koliko ima ukupno
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad(): # Ne racuna gradijente, samo racuna f(x)
            for x, y in val_loader:
                x = x.view(x.size(0), -1).to(device)
                y = y.to(device)

                outputs = model(x)
                _, preds = torch.max(outputs, 1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_state, best_val_acc
