from scipy import stats

class StatisticalValidator:
    @staticmethod
    def verify_repair_significance(pre_repair_accs, post_repair_accs):
        """Performs a paired t-test to calculate statistical confidence metrics."""
        t_stat, p_value = stats.ttest_rel(post_repair_accs, pre_repair_accs)

        # Calculate Cohen's d for effect sizing
        diff = np.array(post_repair_accs) - np.array(pre_repair_accs)
        cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

        is_significant = p_value < 0.001

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d_effect_size": cohen_d,
            "statistically_validated": is_significant
        }
