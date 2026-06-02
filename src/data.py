def generate_data(num_samples=1000):
    X = torch.randn(num_samples, 20)
    # Target function: simple decision boundary
    y = (X[:, 0] + X[:, 1] > 0).long()
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Generate baseline training and curation dataloaders
train_loader = generate_data(800)
clean_heal_loader = generate_data(200)

# Simulate an anomaly batch (adversarial attack / mislabeled noise)
X_anom = torch.randn(32, 20)
# Only poison the features that actually matter to the decision boundary (features 0 and 1)
X_anom[:, 0] = X_anom[:, 0] * 25.0
X_anom[:, 1] = X_anom[:, 1] * -25.0

# Mislabeled target data
y_anom = torch.randint(0, 2, (32,))   # Totally random garbage labels
anomaly_batch = {"x": X_anom, "y": y_anom}
print("✅ Datasets generated. Anomaly trigger vector prepared.")
