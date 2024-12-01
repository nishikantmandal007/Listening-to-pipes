import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from models import MultimodalSensorClassifier  # Import the trained model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_test_data(pressure_path, hydrophone_path, label_path):
    """
    Load the preprocessed test tensor data.
    """
    pressure_data = torch.load(pressure_path)  # Tensor of shape [N, 1, time_steps]
    hydrophone_data = torch.load(hydrophone_path)  # Tensor of shape [N, 1, time_steps]
    labels = torch.load(label_path)  # Tensor of shape [N]
    return pressure_data, hydrophone_data, labels

# Create DataLoader for the test set
def create_test_dataloader(pressure_data, hydrophone_data, labels, batch_size=16):
    """
    Create DataLoader for testing.
    """
    dataset = TensorDataset(pressure_data, hydrophone_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model
def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set and compute various metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for pressure_batch, hydrophone_batch, labels_batch in test_loader:
            pressure_batch, hydrophone_batch, labels_batch = pressure_batch.to(device), hydrophone_batch.to(device), labels_batch.to(device)
            
            outputs = model(pressure_batch, hydrophone_batch)
            probs = outputs.squeeze()  # Predicted probabilities
            preds = torch.round(probs)  # Binary predictions (0 or 1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels_batch.cpu())

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, auc, conf_matrix, all_labels, all_probs

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plot the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.0
    for i, j in enumerate(range(conf_matrix.shape[0])):
        for k in range(conf_matrix.shape[1]):
            plt.text(k, i, format(conf_matrix[i, k], "d"),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, k] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# Plot ROC Curve
def plot_roc_curve(labels, probs):
    """
    Plot the ROC curve.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc_score(labels, probs):.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # Load test data
    pressure_test, hydrophone_test, labels_test = load_test_data("pressure_test.pt", "hydrophone_test.pt", "labels_test.pt")
    
    # Create test DataLoader
    test_loader = create_test_dataloader(pressure_test, hydrophone_test, labels_test, batch_size=16)

    # Load the best model
    model = MultimodalSensorClassifier()
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)

    print("\nEvaluating the model on the test set...")
    accuracy, precision, recall, f1, auc, conf_matrix, all_labels, all_probs = evaluate_model(model, test_loader)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, classes=["No Leak", "Leak"])

    # Plot ROC curve
    plot_roc_curve(all_labels, all_probs)
