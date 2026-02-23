import os
import torch

from src.layerwise_train import layerwise_training


def ensure_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def main():
    print("=== Forward Thinking Layer-wise Training (MNIST) ===")

    ensure_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # pokreni trening
    best_hidden_dims, best_val_acc, test_acc, best_state = layerwise_training(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        max_layers=6,
        device=device
    )

    # --- SNIMANJE MODELA ---
    model_path = "models/best_model.pth"
    torch.save(
        {
            "hidden_dims": best_hidden_dims,
            "state_dict": best_state,
            "val_acc": best_val_acc,
            "test_acc": test_acc,
        },
        model_path
    )

    # --- SNIMANJE REZULTATA ---
    summary_path = "results/summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Forward Thinking Layer-wise Training (MNIST)\n")
        f.write("-------------------------------------------\n")
        f.write(f"Best hidden dims: {best_hidden_dims}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.6f}\n")
        f.write(f"Test accuracy: {test_acc:.6f}\n")
        f.write(f"Saved model: {model_path}\n")

    print("\nTraining finished.")
    print(f"Model saved to: {model_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
