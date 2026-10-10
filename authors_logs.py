# Скачиваем логи обучения авторов (папка Results/ в их репозитории) и оставляем только кривые точности на тесте
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
URL = "https://raw.githubusercontent.com/lnsmith54/super-convergence/master/Results/"
LOGS = { # имя моего запуска -> имя лога, FigN и TabN в названиях это номера рисунков из старой версии статьи
    "range_it5000": "range3Iter5kFig2b",
    "range_it20000": "range3Iter20kFig2b",
    "range_it100000": "range3Iter100kFig2b",
    "clr_n50000": "clr3SS5kFig1a",
    "step_n50000": "lr35Fig1a",
    "clr_wd1e-3_n50000": "clr3SS5kWD-3Fig11",
    "clr_wd1e-5": "clr3SS5kWD-5Fig11",
    "clr_wd1e-6": "clr3SS5kWD-6Fig11",
    **{f"clr_mom{m}": f"clr3SS5kMom{m[2:]}Fig11" for m in ["0.8", "0.85", "0.95"]},
    **{f"clr_n{n}": f"clr3SS5kTr{n // 1000}kTab1" for n in [10000, 20000, 30000, 40000]},
    **{f"step_n{n}": f"lr35Tr{n // 1000}kTab1" for n in [10000, 20000, 30000, 40000]},
    "clr_resnet20": "clr3SS5kRes20Tab1", # ResNet-20/110 (Fig. 4b): в логах CLR stepsize 5000, то есть 10k итераций, а не 20k как в статье
    "step_resnet20": "lr35Res20Fig6b",
    "clr_resnet110": "clr3SS5kRes110Tab1", # у ResNet-110 батч 8 x 100
    "step_resnet110": "lr35Res110Fig6b",
    "clr_cifar100": "clr3SS5kCifar100Fig8", # этого нет в моих запусках: CIFAR-100 (Fig. 10b) и другие оптимизаторы (Fig. 9), везде ResNet-56
    "step_cifar100": "lr35Cifar100Fig8",
    "clr_nesterov": "clr3SS5kNestFig9",
    "step_nesterov": "lr35NestFig9",
    **{f"step_{opt.lower()}": f"lr35{opt}Fig9" for opt in ["AdaDelta", "AdaGrad", "Adam"]},
}

if __name__ == "__main__":
    curves = {}
    for name, log_name in LOGS.items():
        text = urllib.request.urlopen(URL + log_name).read().decode("utf-8", "ignore")
        iters = [int(i) for i in re.findall(r"Iteration (\d+), Testing net", text)] # номера итераций, на которых был тест
        acc = [float(a) for a in re.findall(r"Test net output #0: accuracy = ([\d.]+)", text)] # точность на тесте из строк лога
        curves[name] = {"iter": iters[:len(acc)], "test_acc": acc}
        print(f"{name:20s} {log_name:22s} final test acc {acc[-1]:.4f}, best {max(acc):.4f}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "authors_curves.json", "w") as f:
        json.dump(curves, f)
