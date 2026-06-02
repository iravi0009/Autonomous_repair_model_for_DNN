class PipelineOrchestrator:
    def __init__(self, model):
        self.model = model
        self.baseline_norms = {}
        self.severity_map = {"Low": 2, "Medium": 4, "High": 8}

    def profile_baseline(self, clean_loader, criterion):
        """Captures average normal gradients to set immune threshold profile."""
        self.model.train()
        temp_norms = {name: [] for name, _ in self.model.named_parameters() if _.requires_grad}

        for X, y in clean_loader:
            self.model.zero_grad()
            loss = criterion(self.model(X), y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    temp_norms[name].append(param.grad.norm().item())

        self.baseline_norms = {k: np.mean(v) for k, v in temp_norms.items()}
        print(f"📊 Profiled Baseline Norms: {self.baseline_norms}")

    def analyze_gradients(self):
        """Stage 1 & 2: Evaluates deviations and isolates hot layers."""
        suspicious_layers = []
        total_deviation = 0.0
        current_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                curr_norm = param.grad.norm().item()
                current_norms[name] = curr_norm
                base_norm = self.baseline_norms.get(name, 1.0)

                # Deviation coefficient
                deviation = abs(curr_norm - base_norm) / (base_norm + 1e-8)
                if deviation > 2.5:
                    suspicious_layers.append(name)
                    total_deviation += deviation

        return current_norms, total_deviation, suspicious_layers

    def estimate_severity(self, total_deviation, layers):
        """Stage 3: Severity Estimation & LoRA Rank Mapping."""
        if total_deviation < 2.0: return "Low", self.severity_map["Low"]
        elif total_deviation < 8.0: return "Medium", self.severity_map["Medium"]
        else: return "High", self.severity_map["High"]
