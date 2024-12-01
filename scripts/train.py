import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models import MultimodalSensorClassifier  # Import your model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(pressure_path, hydrophone_path, label_path):
    """
    Load the preprocessed tensor data.
    """
    pressure_data = torch.load(pressure_path)  # Tensor of shape [N, 1, time_steps]
    hydrophone_data = torch.load(hydrophone_path)  # Tensor of shape [N, 1, time_steps]
    labels = torch.load(label_path)  # Tensor of shape [N]
    return pressure_data, hydrophone_data, labels

# Split the data into training, validation, and test sets
def split_data(pressure_data, hydrophone_data, labels, val_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(pressure_data, labels, test_size=val_size+test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=42)

    hydrophone_X_train, hydrophone_X_temp, _, _ = train_test_split(hydrophone_data, hydrophone_data, test_size=val_size+test_size, random_state=42)
    hydrophone_X_val, hydrophone_X_test, _, _ = train_test_split(hydrophone_X_temp, hydrophone_X_temp, test_size=test_size/(val_size+test_size), random_state=42)

    return X_train, X_val, X_test, hydrophone_X_train, hydrophone_X_val, hydrophone_X_test, y_train, y_val, y_test

# Create DataLoader
def create_dataloader(pressure_data, hydrophone_data, labels, batch_size=16):
    """
    Create DataLoader for training.
    """
    dataset = TensorDataset(pressure_data, hydrophone_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model and evaluate on validation set.
    """
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for pressure_batch, hydrophone_batch, labels_batch in train_loader:
            pressure_batch, hydrophone_batch, labels_batch = pressure_batch.to(device), hydrophone_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(pressure_batch, hydrophone_batch)
            loss = criterion(outputs.squeeze(), labels_batch.float())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")  # Save best model

# Evaluate the model on a given dataset
def evaluate_model(model, val_loader):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pressure_batch, hydrophone_batch, labels_batch in val_loader:
            pressure_batch, hydrophone_batch, labels_batch = pressure_batch.to(device), hydrophone_batch.to(device), labels_batch.to(device)
            outputs = model(pressure_batch, hydrophone_batch)
            preds = torch.round(outputs.squeeze())  # Round to 0 or 1 for binary classification
            
            all_preds.append(preds.cpu())
            all_labels.append(labels_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy * 100

def cross_validate(model, pressure_data, hydrophone_data, labels, num_epochs=10, batch_size=16, num_splits=5):
    """
    Perform cross-validation to evaluate the model's generalization ability.
    """
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(pressure_data)):
        print(f"\nFold {fold+1}/{num_splits}")
        
        train_data = pressure_data[train_idx]
        val_data = pressure_data[val_idx]
        train_hydrophone_data = hydrophone_data[train_idx]
        val_hydrophone_data = hydrophone_data[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        train_loader = create_dataloader(train_data, train_hydrophone_data, train_labels, batch_size)
        val_loader = create_dataloader(val_data, val_hydrophone_data, val_labels, batch_size)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

        # Evaluate the model on the validation set
        fold_accuracy = evaluate_model(model, val_loader)
        fold_accuracies.append(fold_accuracy)
        
    print(f"\nCross-validation results:")
    print(f"Mean Accuracy: {sum(fold_accuracies)/num_splits:.2f}%")

if __name__ == "__main__":
    # Load data
    pressure_data, hydrophone_data, labels = load_data("pressure_data.pt", "hydrophone_data.pt", "labels.pt")
    
    # Split data into training, validation, and test sets
    X_train, X_val, X_test, hydrophone_X_train, hydrophone_X_val, hydrophone_X_test, y_train, y_val, y_test = split_data(pressure_data, hydrophone_data, labels)

    # Create DataLoaders
    train_loader = create_dataloader(X_train, hydrophone_X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, hydrophone_X_val, y_val, batch_size=16)

    # Initialize model
    model = MultimodalSensorClassifier()

    # Perform cross-validation
    cross_validate(model, pressure_data, hydrophone_data, labels, num_epochs=10, batch_size=16, num_splits=5)
    
    # Train the model on the full training data and evaluate
    train_model(model, train_loader, val_loader, criterion=nn.BCEWithLogitsLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-4), num_epochs=10)

    # Final evaluation on the test set
    test_loader = create_dataloader(X_test, hydrophone_X_test, y_test, batch_size=16)
    print("\nFinal Test Set Evaluation:")
    evaluate_model(model, test_loader)
