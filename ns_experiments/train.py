import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from fno import FNOZongyi2DBlock
from refno import ReFNN2d
from datasets import KolmDataset

def train_one_epoch(model, optimizer, train_data, scaler=None):
    model.train()
    with torch.autocast('cuda',  enabled=False):
        train_loss = 0
        avg_train_loss = 0
        pb = tqdm(train_data)
        for x, y in pb:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            pred_y = model(x)
            loss = ((y - pred_y)**2).mean() * 1000
            # Moving average in bar
            train_loss += loss.detach().item()
            avg_train_loss += (loss.detach().item() / 1000) / (len(train_data)/train_data.batch_size)
            train_loss /= 2
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pb.set_description(f"Loss: {train_loss} {scaler.get_scale()}")
            else:
                loss.backward()
                optimizer.step()
                pb.set_description(f"Loss: {train_loss} no")
    return avg_train_loss

def validate_one_epoch(model, valid_data):
    with torch.inference_mode():
        valid_loss = 0
        for x, y in tqdm(valid_data):
            x, y = x.cuda(), y.cuda()
            pred_y = model(x)
            loss = ((y - pred_y)**2).mean()
            valid_loss += loss.detach() / (len(valid_data)/valid_data.batch_size)
        print('valid loss:', valid_loss)
        return valid_loss


if __name__ == '__main__':

    exp = 'fno'
    scaler = torch.cuda.amp.GradScaler()
    # scaler = None
    batch_size = 116
    epochs = 10
    modes = 16
    v = 4
    mode_options = [ 16, 24]
    exps = [
            'stab_nl',
            'stab',
            'fno',
            ]
    vs = [1, 2, 3, 4, 5, 6]
    for v in vs:
        for exp in exps:
            for modes in mode_options:
                if exp == 'fno':
                    model = FNOZongyi2DBlock(modes, modes, 128, input_dim=2).cuda()
                elif exp == 'stab':
                    model = ReFNN2d(128, 2, ratio=modes/32).cuda()
                elif exp == 'stab_nl':
                    model = ReFNN2d(128, 2, ratio=modes/32, nonlinear=True).cuda()
                print(exp, modes, 'model init. params:', sum([p.numel() for p in model.parameters()]))
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
                train_data = torch.utils.data.DataLoader(KolmDataset('train'), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
                valid_data = torch.utils.data.DataLoader(KolmDataset('valid'), batch_size=256, shuffle=True)
                for i in range(epochs):
                    print('Epoch %s' % i)
                    print('Train')
                    train_loss = train_one_epoch(model, optimizer, train_data, scaler)
                    scheduler.step()
                    print('Valid')
                    valid_loss = validate_one_epoch(model, valid_data)
                    # break
                    torch.save({
                                'epoch': i,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': valid_loss,
                                'train_loss': train_loss,
                                }, f'{exp}_{modes}_{v}.ckpt')