import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, loader, epochs=50, lr=1e-3, device="cuda", lambda_smooth=0.0):
    opt = optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss()
    model.train()
    for ep in range(epochs):
        running = 0.0
        for v2, d in loader:
            v2, d = v2.to(device), d.to(device)
            pred = model(v2)
            loss = l1(pred, d)
            if lambda_smooth > 0:
                grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                smooth = grad_x.abs().mean() + grad_y.abs().mean()
                loss = loss + lambda_smooth * smooth
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * v2.size(0)
        avg = running / len(loader.dataset)
        print(f"Epoch {ep+1}/{epochs}  Loss: {avg:.6f}")
    return model

def sliding_predict_residual(model, arr2, tile=256, overlap=32, device="cuda", mean=0.0, std=1.0):
    H, W = arr2.shape
    out = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                hh = min(tile, H - y)
                ww = min(tile, W - x)
                patch = arr2[y:y+hh, x:x+ww].astype(np.float32)
                pad = np.zeros((tile, tile), dtype=np.float32)
                pad[:hh, :ww] = (patch - mean) / std
                inp = torch.tensor(pad).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(inp).squeeze().cpu().numpy()
                pred = pred[:hh, :ww]
                out[y:y+hh, x:x+ww] += pred
                weight[y:y+hh, x:x+ww] += 1.0
    weight[weight == 0] = 1.0
    return out / weight
