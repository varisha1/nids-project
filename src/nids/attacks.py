import numpy as np
import torch
import torch.nn.functional as F
from .models import make_surrogate


def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    (grad,) = torch.autograd.grad(loss, x_adv)
    return (x + eps * grad.sign()).detach()


def pgd(model, x, y, eps, alpha=0.01, steps=10):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).detach()
    return x_adv


def batched_attack(attack_fn, model, X, y, device, bs=10000, **kw):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(y[i:i + bs]).to(device)
        out.append(attack_fn(model, xb, yb, **kw).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_surrogate(target_predict, X_ref, budget, device, seed=42, noise_std=0.1, epochs=20):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_ref), size=min(budget, len(X_ref)), replace=False)
    Xs = (X_ref[idx] + rng.normal(0, noise_std, X_ref[idx].shape)).astype(np.float32)
    ys = target_predict(Xs).astype(np.int64)
    sur = make_surrogate(Xs.shape[1]).to(device)
    opt = torch.optim.Adam(sur.parameters(), lr=1e-3)
    X_t, y_t = torch.from_numpy(Xs).to(device), torch.from_numpy(ys).to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), device=device)
        for i in range(0, len(X_t), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            F.cross_entropy(sur(X_t[b]), y_t[b]).backward()
            opt.step()
    return sur