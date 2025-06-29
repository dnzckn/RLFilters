
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import FilterDataset
from src.model import EM_Emulator

# --- Configuration ---
CSV_FILE = "data/synthetic_data.csv"
GRID_SIZE = 16
NUM_FREQ_POINTS = 100
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "models/em_emulator.pth"

def train_model():
    """Trains the EM emulator model."""
    # -- 1. Load Data --
    dataset = FilterDataset(csv_file=CSV_FILE, grid_size=GRID_SIZE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -- 2. Initialize Model, Loss, and Optimizer --
    model = EM_Emulator(grid_size=GRID_SIZE, num_freq_points=NUM_FREQ_POINTS).float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -- 3. Training Loop --
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            layouts = sample['layout'].float()
            s_params = sample['s_params'].float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(layouts)
            loss = criterion(outputs, s_params)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # -- Validation --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sample in val_loader:
                layouts = sample['layout'].float()
                s_params = sample['s_params'].float()
                outputs = model(layouts)
                loss = criterion(outputs, s_params)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    print("Finished Training")

    # -- 4. Save Model --
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
