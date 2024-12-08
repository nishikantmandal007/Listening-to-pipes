from tabulate import tabulate

# Define the table data
table = [
    ["Loss", 0.2345],
    ["Accuracy", 0.9123],
    ["Precision", 0.8765],
    ["Recall", 0.8345],
    ["F1 Score", 0.8543]
]

# Define the headers
headers = ["Metric", "Value"]

# Print the table
print("Leak Detection Model Evaluation Metrics: ROC -AUC ")
print(tabulate(table, headers, tablefmt="grid"))