import torch
import lpips

_lpips_fn = None

def lpips_score(pred, gt):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="vgg").cuda()
    # pred/gt: [1, 3, H, W] in [0, 1]
    pred_scaled = pred * 2.0 - 1.0
    gt_scaled = gt * 2.0 - 1.0
    with torch.no_grad():
        return _lpips_fn(pred_scaled, gt_scaled).item()
