# Оценка оптимального LR по весам на трех итерациях подряд (раздел 4, Fig. 3a):
#     eps* = eps * sum|w1 - w0| / sum|2 * w1 - w0 - w2|, потом сглаживание с alpha = 0.1
# В скрипте авторов (L2TermComputation3.py) сумма идет по net.params из Caffe, а у слоев BatchNorm
# там лежат не веса, а накопленные суммы статистик. Поэтому оценка считается два раза: только по весам
# и по весам вместе с суммами BN ровно как в Caffe, для настроек BN 0.95 и 0.999
import json

import torch
import torch.nn.functional as F

from data import load_cifar10, train_batches
from model import ResNet
from train import ROOT


def lr_at(it, schedule):
    if schedule == "const": # постоянный LR 0.1 для сравнения
        return 0.1

    t = it / 5000 # CLR 0.1-3 со stepsize 5000, как в статье
    return 0.1 + 2.9 * (1 - abs(t - 1))


def flat(tensors):
    return torch.cat([t.detach().float().flatten() for t in tensors])


def run(schedule, maf, x_train, y_train, iters=300, alpha=0.1):
    torch.manual_seed(0)
    model = ResNet(56).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9, weight_decay=1e-4)
    batches = train_batches(x_train, y_train, 1000)

    params = list(model.parameters())
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    s = 0.0 # счетчик s из Caffe
    history = []
    curves = {"weights": [], "weights+bn_stats": []}

    for it in range(iters):
        lr = lr_at(it, schedule)
        for group in opt.param_groups:
            group["lr"] = lr
        s = maf * s + 1 # BatchNorm как в Caffe, так же как в train.py с --bn-maf
        for bn in bns:
            bn.momentum = 1 / s

        x, y = next(batches)
        with torch.autocast("cuda", torch.bfloat16):
            out = model(x.contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        caffe_bn = []
        for bn in bns: # у BN в Caffe три блоба: сумма средних, сумма дисперсий и сам s
            caffe_bn += [bn.running_mean * s, bn.running_var * s, torch.full((1,), s, device="cuda")]
        history.append({"weights": flat(params), "weights+bn_stats": flat(params + caffe_bn)})
        if len(history) == 3:
            for key, curve in curves.items():
                w0, w1, w2 = (h[key] for h in history)
                est = (lr * (w1 - w0).abs().sum() / (2 * w1 - w0 - w2).abs().sum()).item() # формула 8 сразу по всем весам, как у авторов
                curve.append(est if not curve else alpha * est + (1 - alpha) * curve[-1]) # сглаживание как в статье
            history.pop(0)
    return curves


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    x_train, y_train, _, _ = load_cifar10(ROOT / "data")
    result = {f"{schedule}_maf{maf}": run(schedule, maf, x_train, y_train) for schedule in ["const", "clr"] for maf in [0.95, 0.999]}
    for name, curves in result.items():
        for key, est in curves.items():
            print(f"{name:14s} {key:16s} estimate at the end {est[-1]:.2f}, range {min(est):.2f}-{max(est):.2f}")
    with open(ROOT / "results" / "lr_estimate.json", "w") as f:
        json.dump(result, f)
