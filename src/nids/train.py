import torch
import torch.nn.functional as F
from .attacks import pgd


def _loader(X, y, bs, device):
    X_t, y_t = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    perm = torch.randperm(len(X_t), device=device)
    for i in range(0, len(X_t), bs):
        b = perm[i:i + bs]
        yield X_t[b], y_t[b]


def train_standard(model, X, y, epochs, bs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in _loader(X, y, bs, device):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


def train_adversarial(model, X, y, epochs, bs, lr, eps_max, lam_align, lam_smooth,
                      pgd_alpha, pgd_steps, device):
    """Curriculum adversarial training: eps_e = eps_max * e / E.
    L = CE(clean) + CE(adv) + lam_align * L_align + lam_smooth * L_smooth."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(1, epochs + 1):
        eps_e = eps_max * e / epochs
        for xb, yb in _loader(X, y, bs, device):
            model.eval()
            x_adv = pgd(model, xb, yb, eps_e, pgd_alpha, pgd_steps)
            model.train()

            xc = xb.clone().requires_grad_(True)
            xa = x_adv.clone().requires_grad_(True)
            fc, fa = model.features(xc), model.features(xa)
            ce_c = F.cross_entropy(model.head(fc), yb, reduction="none")
            ce_a = F.cross_entropy(model.head(fa), yb, reduction="none")
            (gc,) = torch.autograd.grad(ce_c.sum(), xc, create_graph=True)
            (ga,) = torch.autograd.grad(ce_a.sum(), xa, create_graph=True)

            l_align = ((gc - ga) ** 2).sum(1).mean()
            l_smooth = ((fc - fa) ** 2).sum(1).mean()
            loss = ce_c.mean() + ce_a.mean() + lam_align * l_align + lam_smooth * l_smooth

            opt.zero_grad()
            loss.backward()
            opt.step()
    return model