print("🏋️‍♂️ Phase 1: Training Baseline Model...")
epochs_for_baseline = 5 # Give it enough time to learn the basic patterns

for epoch in range(epochs_for_baseline):
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        base_model.train()
        optimizer.zero_grad()
        outputs = base_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    acc = 100. * correct / total
    print(f"   ↳ Epoch {epoch+1}/{epochs_for_baseline} - Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

print("✅ Baseline model trained and stabilized.")
