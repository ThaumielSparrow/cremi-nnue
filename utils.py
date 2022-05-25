import torch
import torchvision
from dataload import LoadData
from torch.utils.data import DataLoader

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('=> Current state saved')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('=> Loaded state')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_ds = LoadData(
        em_dir=train_dir,
        seg_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = LoadData(
        em_dir=val_dir,
        seg_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds*y).sum()) / (preds+y).sum() + 1e-8

    print(f'{num_correct}/{num_pixels} correct\nAccuracy: {((num_correct/num_pixels)*100):.2f}%')
    print(f'Dice score: {dice_score/len(loader):.3f}')

    model.train()

def save_predictions_as_imgs(loader, model, dir='saved_images/', device='cuda'):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{dir}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{dir}/{idx}.png')
    
    model.train()
