from piq import LPIPS

lpips = LPIPS(replace_pooling=True, reduction="none")


def calculate_lpips(x, x_hat):
    """Create LPIPS loss calculator"""
    return lpips(x, x_hat).mean()
