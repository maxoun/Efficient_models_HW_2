import torch
import torch.nn.functional as F
import torchvision


def load_cifar10(root="data", train_size=50000, device="cuda"):
    train = torchvision.datasets.CIFAR10(root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True)

    x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).float()
    x_test = torch.from_numpy(test.data).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(train.targets)
    y_test = torch.tensor(test.targets)

    mean = x_train.mean(0) # как в конфиге авторов: вычитаем среднюю картинку и больше ничего
    x_train -= mean
    x_test -= mean

    if train_size < len(x_train):
        idx = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(0))[:train_size] # подвыборка всегда одна и та же, чтобы все расписания видели одни и те же картинки
        x_train, y_train = x_train[idx], y_train[idx]

    return [t.to(device) for t in (x_train, y_train, x_test, y_test)]


def load_mnist(root="data", train_size=60000, device="cuda"):
    train = torchvision.datasets.MNIST(root, train=True, download=True)
    test = torchvision.datasets.MNIST(root, train=False, download=True)

    x_train = train.data.unsqueeze(1).float() / 256 # как в примере LeNet из Caffe: пиксели умножаем на 1/256, среднее не вычитаем
    x_test = test.data.unsqueeze(1).float() / 256
    y_train, y_test = train.targets, test.targets

    if train_size < len(x_train):
        idx = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(0))[:train_size]
        x_train, y_train = x_train[idx], y_train[idx]

    return [t.to(device) for t in (x_train, y_train, x_test, y_test)]


def random_crop(x, pad=4): # обычная аугментация для CIFAR: дополняем картинку нулями по краям и вырезаем случайный кусок 32x32
    n, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=x.device)
    rows = dy + torch.arange(h, device=x.device).view(1, 1, h, 1)
    x = x.gather(2, rows.expand(n, c, h, w + 2 * pad))
    cols = dx + torch.arange(w, device=x.device).view(1, 1, 1, w)
    return x.gather(3, cols.expand(n, c, h, w))


def train_batches(x, y, batch_size, crop=False, flip=True): # бесконечный поток перемешанных батчей, в статье из аугментаций только flip
    while True:
        perm = torch.randperm(len(x), device=x.device) # каждую эпоху новый порядок картинок
        for i in range(0, len(x) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            if flip: # отражаем по горизонтали случайную половину батча
                mask = torch.rand(len(idx), 1, 1, 1, device=x.device) < 0.5
                xb = torch.where(mask, xb.flip(3), xb)
            yield (random_crop(xb) if crop else xb), y[idx]
