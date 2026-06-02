class PatchManager:
    @staticmethod
    def deploy_patch(model, rank, target_layers):
        """Dynamically injects a focused micro-LoRA patch configuration."""
        # Strip strings to map clean sub-modules for PEFT targeting
        # Modify the logic to map 'bnX' layers to their corresponding 'fcX' layers for patching.
        modules_to_patch_raw = [layer.split('.')[0] for layer in target_layers]
        final_modules_to_patch = []
        for module_name in modules_to_patch_raw:
            if "fc" in module_name:
                final_modules_to_patch.append(module_name)
            elif "bn" in module_name:
                try:
                    # Extract number from 'bnX' and map to 'fcX'
                    num = int(module_name.replace('bn', ''))
                    final_modules_to_patch.append(f"fc{num}")
                except ValueError:
                    # If parsing fails, skip or handle differently if other bn names exist
                    pass
        modules_to_patch = list(set(final_modules_to_patch))

        config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=modules_to_patch,
            lora_dropout=0.05,
            bias="none"
        )
        print(f"🛠️ Injecting LoRA patch (Rank {rank}) into targeted modules: {modules_to_patch}")
        return get_peft_model(model, config)

    @staticmethod
    def heal(patched_model, clean_loader, criterion, epochs=15, accumulation_steps=4):
        """Fine-tunes the micro-patch exclusively on clean datasets."""
        optimizer = optim.AdamW(patched_model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Initialize the scheduler ONCE before the epochs start
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        patched_model.train()
        total_loss = 0.0 # Initialize total_loss here

        for i, (X, y) in enumerate(clean_loader):
                outputs = patched_model(X)
                loss = criterion(outputs, y)

                # Normalize the loss based on accumulation steps
                loss = loss / accumulation_steps
                loss.backward()

                # Only step the optimizer after accumulating enough gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(clean_loader):
                    torch.nn.utils.clip_grad_norm_(patched_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad() # Reset after stepping

                total_loss += loss.item() * accumulation_steps

            # Step the scheduler ONCE per epoch (after all batches have been processed)
        scheduler.step()

        print(f"🩹 Healing complete. Post-repair convergence loss: {total_loss/len(clean_loader):.4f}")
        return patched_model
