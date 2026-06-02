import time
import psutil
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class MetricsEngine:
    @staticmethod
    def evaluate_task_performance(model, data_loader, criterion, device):
        """Calculates precise classification performance, latency, and memory footprint."""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        # Track inference speed
        start_time = time.perf_counter()
        process = psutil.Process()
        start_mem = process.memory_info().rss / (1024 ** 2) # MB

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / (1024 ** 2) # MB

        # Calculate derived statistical metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )

        metrics = {
            "accuracy": (correct / total) * 100,
            "avg_loss": test_loss / len(data_loader),
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100,
            "latency_ms_per_batch": ((end_time - start_time) / len(data_loader)) * 1000,
            "memory_delta_mb": max(0.0, end_mem - start_mem)
        }
        return metrics

class EfficiencyProfiler:
    @staticmethod
    def calculate_parameter_efficiency(model):
        """Measures parameter reduction provided by targeted LoRA allocation."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        reduction_percentage = (1.0 - (trainable_params / total_params)) * 100
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_reduction_rate": reduction_percentage
        }

    @staticmethod
    def calculate_repair_speedup(time_full_retrain, time_healer_patch):
        """Computes the operational speedup factor achieved by the framework."""
        speedup_factor = time_full_retrain / (time_healer_patch + 1e-8)
        time_saved_pct = ((time_full_retrain - time_healer_patch) / time_full_retrain) * 100

        return {
            "speedup_multiplier": speedup_factor,
            "compute_time_saved_percentage": time_saved_pct
        }
