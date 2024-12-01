import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import LeakDetectionModel

# Dataset Class
class LeakDataset(Dataset):
    def __init__(self, pressure_path, hydrophone_path, label, max_len=None):
        # Load tensors
        self.pressure_data = torch.load(pressure_path)
        self.hydrophone_data = torch.load(hydrophone_path)
        self.label = torch.tensor([label] * len(self.pressure_data))

        # Determine maximum length if not provided
        self.max_len = max_len or max(len(self.pressure_data), len(self.hydrophone_data))

        # Synchronize hydrophone data with pressure data based on index ratio
        self.hydrophone_data = self._sync_to_pressure(self.hydrophone_data, len(self.pressure_data))

        # Pad or truncate to fixed length
        self.pressure_data = self._pad_or_truncate(self.pressure_data, self.max_len)
        self.hydrophone_data = self._pad_or_truncate(self.hydrophone_data, self.max_len)

    def _sync_to_pressure(self, hydrophone_data, target_length):
        """
        Synchronize hydrophone data to match the pressure data length.
        If hydrophone data is longer, map to nearest pressure data index.
        """
        hydro_len = len(hydrophone_data)
        if hydro_len == target_length:
            return hydrophone_data

        # Create a mapping from pressure indices to hydrophone indices
        mapped_indices = torch.linspace(0, hydro_len - 1, steps=target_length).round().long()
        return hydrophone_data[mapped_indices]

    def _pad_or_truncate(self, data, target_length):
        # Pad with zeros or truncate to target_length
        if len(data) > target_length:
            return data[:target_length]
        elif len(data) < target_length:
            padding = torch.zeros(target_length - len(data), *data.shape[1:])
            return torch.cat((data, padding), dim=0)
        return data

    def __len__(self):
        return len(self.pressure_data)  # Base length on pressure data to avoid mismatch

    def __getitem__(self, idx):
        pressure = self.pressure_data[idx]
        hydrophone = self.hydrophone_data[idx]

        # Ensure hydrophone has the correct dimensions
        if hydrophone.ndim == 1:
            hydrophone = hydrophone.unsqueeze(0)  # Add a channel dimension if missing

        label = self.label[idx]
        return pressure, hydrophone, label


# Custom collate function for DataLoader
def custom_collate_fn(batch):
    pressures, hydrophones, labels = zip(*batch)
    pressures = torch.stack(pressures)
    hydrophones = torch.stack(hydrophones)
    labels = torch.tensor(labels)
    return pressures, hydrophones, labels


# Load Dataset
def load_dataset():
    leak_pressure_path = 'data/processed/Dynamic Pressure Sensor/p_leak.pt'
    leak_hydrophone_path = 'data/processed/Hydrophone/h_leak.pt'
    no_leak_pressure_path = 'data/processed/Dynamic Pressure Sensor/p_noleak.pt'
    no_leak_hydrophone_path = 'data/processed/Hydrophone/h_noleak.pt'

    # Define a fixed length for alignment (choose the max length across datasets)
    max_length = 25  # Adjust this value if required
    leak_data = LeakDataset(leak_pressure_path, leak_hydrophone_path, label=1, max_len=max_length)
    noleak_data = LeakDataset(no_leak_pressure_path, no_leak_hydrophone_path, label=0, max_len=max_length)

    # Combine datasets for pressure and hydrophone data
    full_dataset = torch.utils.data.ConcatDataset([leak_data, noleak_data])

    # Split into train (90%) and validation (10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset


# Training Function
def train_model(model, train_loader, val_loader, device, hyperparams):
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(hyperparams['epochs']):
        model.train()
        train_loss = 0.0
        for pressure, hydrophone, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']}"):
            pressure, hydrophone, labels = pressure.to(device), hydrophone.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(pressure, hydrophone)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_outputs, val_labels = 0.0, [], []
        with torch.no_grad():
            for pressure, hydrophone, labels in val_loader:
                pressure, hydrophone, labels = pressure.to(device), hydrophone.to(device), labels.to(device)
                outputs = model(pressure, hydrophone)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))

        # Log metrics
        print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")
        print("Validation Metrics:")
        print(confusion_matrix(val_labels, val_outputs))
        print(classification_report(val_labels, val_outputs))
        print(f"Validation Accuracy: {accuracy_score(val_labels, val_outputs):.4f}")

    return train_losses, val_losses


# Plot Loss Graphs
def plot_losses(train_losses, val_losses, output_dir):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


# Main Function
if __name__ == "__main__":
    hyperparams = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 20
    }

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare Dataset
    train_dataset, val_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update the model initialization to match the hydrophone data shape (e.g., hydrophone_dim=2)
    model = LeakDetectionModel(pressure_dim=25, hydrophone_dim=2).to(device)  # Adjusted hydrophone_dim

    train_losses, val_losses = train_model(model, train_loader, val_loader, device, hyperparams)

    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pth"))
    print("Model and weights saved to", output_dir)

    plot_losses(train_losses, val_losses, output_dir)
