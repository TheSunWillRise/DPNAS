import torch
import opacus.privacy_analysis as tf_privacy

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_renyi_divergence(sample_rate, noise_multiplier, orders=ORDERS):
    rdp = torch.tensor(
        tf_privacy.compute_rdp(
            sample_rate, noise_multiplier, 1, orders
        )
    )
    return rdp


def get_privacy_spent(total_rdp, target_delta=1e-5, orders=ORDERS):
    return tf_privacy.get_privacy_spent(orders, total_rdp, target_delta)


