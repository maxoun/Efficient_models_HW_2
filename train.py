# Обучение сети на CIFAR-10 или MNIST с одним из видов LR со статьм.

#     python train.py --schedule clr --lr-min 0.1 --lr 3 --iters 10000        = super-convergence 
#     python train.py --schedule step --lr 0.35 --iters 80000                 = piecewise constant baseline
#     python train.py --schedule range --lr 3 --iters 5000                    = LR range test 
#     python train.py --schedule onecycle --lr 3 --iters 10000                = OneCycleLR from PyTorch
#     python train.py --schedule cosine --lr 3 --iters 10000 --warmup 0.05    =linear warmup + cosine
#     python train.py --schedule 1cycle --lr-min 0.1 --lr 3 --iters 10000 --cycle-momentum 0.95 0.85       = 1cycle раздел 3 в статье
#     python train.py --dataset mnist --arch lenet --schedule inv --lr 0.01 --batch-size 512 --iters 9961 --wd 5e-4
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_cifar10, load_mnist, train_batches
from model import DenseNet, LeNet, ResNet

ROOT = Path(__file__).parent


def triangle(it, length): # тот самый треугольеый LR, маленький - большой - маленький
    return 1 - abs(2 * it / length - 1)


def get_lr(it, args):
    if args.schedule == "clr":
        t = it / (args.iters / 2) # один треугольный цикл как в Caffe, узнал через Claude
        return args.lr_min + (args.lr - args.lr_min) * (1 - abs(t - 1))
    
    if args.schedule == "step": # ступенчатый LR, бейзлайн
        return args.lr * args.step_gamma ** sum(it >= m for m in args.milestones)
    
    if args.schedule == "range": # LR range test из статьи
        return args.lr * it / args.iters
    
    if args.schedule == "cosine": # разогрев + косинус
        warmup = int(args.warmup * args.iters)
        if it < warmup:
            return args.lr * it / warmup
        
        t = (it - warmup) / (args.iters - warmup)
        return args.lr * (1 + math.cos(math.pi * t)) / 2
    
    if args.schedule == "1cycle": # 1cycle как из статьи
        cycle = round(args.cycle_frac * args.iters)
        if it < cycle:
            return args.lr_min + (args.lr - args.lr_min) * triangle(it, cycle)
        
        t = (it - cycle) / (args.iters - cycle) # хвост после цикла, LR становится очень маленьким
        return args.lr_min * (1 - t + t / args.tail_div)
    
    if args.schedule == "inv": # метод inv из Caffe, тоже Claude подсказал
        return args.lr * (1 + args.inv_gamma * it) ** -args.inv_power
    
    raise ValueError(args.schedule)


def get_momentum(it, args): # тот самый momentum наоборот с LR из презентации
    high, low = args.cycle_momentum
    cycle = round(args.cycle_frac * args.iters)
    return high - (high - low) * triangle(it, cycle) if it < cycle else high


def build_model(args, num_classes): # выбираем сеть по --arch, wide ResNet это тот же ResNet, только с factor=4
    if args.arch == "lenet":
        return LeNet(num_classes)
    
    if args.arch == "densenet":
        return DenseNet(args.depth, num_classes=num_classes, bn_momentum=args.bn_momentum)
    
    return ResNet(args.depth, num_classes, bn_momentum=args.bn_momentum, factor=4 if args.arch == "wide" else 2)


@torch.no_grad()
def evaluate(model, x, y, batch_size=2000): # проверка на тесте кусками по 2000 картинок, без градиентов
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        with torch.autocast("cuda", torch.bfloat16):
            out = model(x[i:i + batch_size]).float()
        loss += F.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


def main(): # аргументы для запуска, а их было много...
    
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", choices=["clr", "step", "range", "onecycle", "cosine", "1cycle", "inv"], required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--lr-min", type=float, default=0.1)
    p.add_argument("--milestones", type=int, nargs="+", default=[50000, 70000])
    p.add_argument("--step-gamma", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--cycle-frac", type=float, default=0.9)
    p.add_argument("--tail-div", type=float, default=1000)
    p.add_argument("--cycle-momentum", type=float, nargs=2, default=None, metavar=("MAX", "MIN"))
    p.add_argument("--inv-gamma", type=float, default=1e-4)
    p.add_argument("--inv-power", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--arch", choices=["resnet", "wide", "densenet", "lenet"], default="resnet")
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--bn-momentum", type=float, default=0.05)
    p.add_argument("--bn-maf", type=float, default=None)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    
    if args.cycle_momentum and args.schedule != "1cycle": # циклический momentum сделан под форму 1cycle, с другими расписаниями он не совпадет с LR, поэтому запрещаем
        p.error("--cycle-momentum works only with --schedule 1cycle")

    args.train_size = args.train_size or {"cifar10": 50000, "mnist": 60000}[args.dataset] # по умолчанию берем весь датасет
    args.depth = args.depth or {"resnet": 56, "wide": 32, "densenet": 40}.get(args.arch) # глубина по умолчанию: 56 у ResNet, 32 у wide, 40 у DenseNet

    name = args.name or f"{args.schedule}_lr{args.lr}_n{args.train_size}_it{args.iters}"
    out_path = ROOT / "results" / f"{name}.json"
    out_path.parent.mkdir(exist_ok=True)

    torch.manual_seed(args.seed) # фиксируем seed, чтобы запуск можно было повторить
    torch.backends.cudnn.benchmark = True # cudnn сам подбирает самые быстрые свертки

    load = load_mnist if args.dataset == "mnist" else load_cifar10
    x_train, y_train, x_test, y_test = load(ROOT / "data", args.train_size) # данные сразу лежат на GPU
    
    x_test = x_test.contiguous(memory_format=torch.channels_last) # channels_last и bf16 ускоряют обучение
    model = build_model(args, int(y_train.max()) + 1).cuda().to(memory_format=torch.channels_last)

    opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=args.momentum, weight_decay=args.wd) # обычный SGD, LR ставим сами на каждой итерации
    batches = train_batches(x_train, y_train, args.batch_size, crop=args.crop, flip=args.dataset == "cifar10") # бесконечный поток батчей, flip только для CIFAR
    scheduler = None

    if args.schedule == "onecycle": # OneCycleLR из PyTorch с настройками по умолчанию
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)] # все слои BatchNorm, нужны для режима как в Caffe
    bn_scale = 0.0 # счетчик s из Caffe

    log = {"args": vars(args), "iter": [], "lr": [], "momentum": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} # все, что сохраняем в results .json
    train_loss = torch.zeros((), device="cuda")
    train_correct = torch.zeros((), device="cuda")
    seen = 0
    start = time.time()

    for it in range(args.iters): # основной цикл обучения
        if scheduler is None: # в начале каждой итерации ставим LR (и momentum) по расписанию
            lr = get_lr(it, args)
            for group in opt.param_groups:
                group["lr"] = lr
                if args.cycle_momentum:
                    group["momentum"] = get_momentum(it, args)
        else: # у onecycle LR меняет сам PyTorch
            lr = opt.param_groups[0]["lr"]
        momentum = opt.param_groups[0]["momentum"]
        if args.bn_maf is not None: # повторяем BatchNorm из Caffe
            bn_scale = args.bn_maf * bn_scale + 1
            for bn in bns:
                bn.momentum = 1 / bn_scale

        x, y = next(batches) # берем следующий батч
        x = x.contiguous(memory_format=torch.channels_last)

        with torch.autocast("cuda", torch.bfloat16): # прямой проход в bf16
            out = model(x)
        loss = F.cross_entropy(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward() # считаем градиенты
        opt.step() # шаг SGD
        if scheduler is not None:
            scheduler.step()

        train_loss += loss.detach() * len(y) # копим loss и точность на обучении между проверками
        train_correct += (out.argmax(1) == y).sum()
        seen += len(y)

        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters: # каждые 100 итераций проверка на тесте, запись в лог и печать
            test_loss, test_acc = evaluate(model, x_test, y_test)
            log["iter"].append(it + 1)
            log["lr"].append(lr)
            log["momentum"].append(momentum)
            log["train_loss"].append(train_loss.item() / seen)
            log["train_acc"].append(train_correct.item() / seen)
            log["test_loss"].append(test_loss)
            log["test_acc"].append(test_acc)
            log["time"] = time.time() - start
            train_loss.zero_()
            train_correct.zero_()
            seen = 0

            print(f"it {it + 1:6d}  lr {lr:.4f}  train loss {log['train_loss'][-1]:.4f} "
                  f"acc {log['train_acc'][-1]:.4f}  test loss {test_loss:.4f} acc {test_acc:.4f}  "
                  f"{log['time']:.0f}s", flush=True)
            
            with open(out_path, "w") as f:
                json.dump(log, f)

            if not math.isfinite(log["train_loss"][-1]): # loss стал NaN или inf, значит обучение разошлось, останавливаемся
                break

    print(f"done {name}: test acc {log['test_acc'][-1]:.4f} (best {max(log['test_acc']):.4f}), "
          f"{log['time'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
