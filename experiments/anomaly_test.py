# Step 2: Inject Anomaly Event
print("\n💥 Event: Poisoned / Out-Of-Distribution Anomaly hit processing pipeline.")
base_model.zero_grad()
anom_loss = criterion(base_model(anomaly_batch["x"]), anomaly_batch["y"])
anom_loss.backward() # Forces massive aberrant gradient updates down parameters

# Step 3: Run Stage 1 & 2 Diagnostics
current_grads, total_deviation, hot_layers = orchestrator.analyze_gradients()

if hot_layers:
    print(f"🚨 ANOMALY detected. Total deviation metrics: {total_deviation:.2f}")
    print(f"🔍 System Diagnostic Report: Hot modules isolated -> {hot_layers}")

    # Step 4: Stage 3 Severity Rating
    severity, assigned_rank = orchestrator.estimate_severity(total_deviation, hot_layers)
    print(f"🎚️ Severity Evaluation: Risk Level [{severity}] -> Assumed Target Rank: {assigned_rank}")

    # Step 5: Stage 4 Activation & Healing Phase
    pre_repair_grads = {k: v for k, v in current_grads.items()} # Snapshot before patch

    patched_model = PatchManager.deploy_patch(base_model, assigned_rank, hot_layers)
    healed_model = PatchManager.heal(patched_model, clean_heal_loader, criterion)

    # Step 6: Stage 5 Validation Verification
    print("\n🛡️ Stage 5: Evaluating Before/After Gradient Variances...")
    healed_model.zero_grad()
    X_val, y_val = next(iter(clean_heal_loader))
    val_loss = criterion(healed_model(X_val), y_val)
    val_loss.backward()

    stable = True
    for name, param in healed_model.named_parameters():
        if "lora" in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"   ↳ Patch Gradient Norm [{name}]: {grad_norm:.5f}")
            if grad_norm > 5.0: stable = False

    if stable:
        print("✅ Delta-Gradient Signature Check passed. System metrics verified as stable.")
        print("💾 Saving active adapter to Google Drive / Checkpoint Safe Bank.")
        healed_model.save_pretrained("checkpoints/patch_bank/active_stable_patch")
    else:
        print("❌ Validation Failed. Active safety rollback protocols required.")
else:
    print("🟢 Gradient checks indicate safe operation boundaries.")
