import torch
from tqdm import tqdm
from utils.history_collect import AverageMeter


def train_epoch(model, loader, device, epoch, optimizer, criterion, scaler, accumulate=1):
    model.train()

    stream = tqdm(loader)

    metric = {
        'loss': AverageMeter(),
        'epoch': epoch + 1,
        'lr': 0
    }

    for i, data in enumerate(stream):
        images, targets = data

        preds = model(images.to(device))

        loss = criterion(preds, targets.to(device))

        scaler.scale(loss).backward()

        # ------------- 梯度累积 -------------
        if (i + 1) % accumulate == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

        metric['loss'].update(loss.detach().item())
        metric['lr'] = optimizer.param_groups[0]['lr']
        stream.set_postfix(**metric)

    return metric


@torch.no_grad()
def val_epoch(model, loader, device, epoch, criterion):
    model.eval()

    stream = tqdm(loader)

    metric = {
        'loss': AverageMeter(),
        'epoch': epoch,
    }

    for i, data in enumerate(stream, start=1):
        images, targets = data

        preds = model(images.to(device))

        loss = criterion(preds, targets.to(device))

        metric['loss'].update(loss.detach().item())
        stream.set_postfix(**metric)

    return metric
