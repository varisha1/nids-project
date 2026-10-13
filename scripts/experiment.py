import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nids.attacks import batched_attack, build_surrogate, fgsm, pgd
from nids.data import load_dataset
from nids.evaluate import metrics, rf_predict, torch_predict
from nids.models import MLP, TransformerNIDS, make_rf
from nids.train import train_adversarial, train_standard
from nids.utils import get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    device = get_device()
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    Xtr, Xte, ytr, yte, names = load_dataset(cfg)
    d = Xtr.shape[1]
    print(f"device={device} features={d} train={len(Xtr)} test={len(Xte)} attack_ratio={ytr.mean():.3f}")

    t, a, ev = cfg["train"], cfg["adv"], cfg["eval"]
    models, rows = {}, []

    for name in cfg["models"]:
        print(f"training {name} ...")
        if name == "rf":
            models[name] = make_rf(cfg["seed"]).fit(Xtr, ytr)
        elif name == "nn":
            models[name] = train_standard(MLP(d), Xtr, ytr, t["epochs"], t["batch_size"], t["lr"], device)
        elif name == "transformer":
            models[name] = train_standard(TransformerNIDS(d), Xtr, ytr, t["epochs"], 1000, t["lr"], device)
        elif name == "adv_nn":
            models[name] = train_adversarial(
                MLP(d), Xtr, ytr, t["epochs"], t["batch_size"], t["lr"], a["eps_max"],
                a["lam_align"], a["lam_smooth"], a["pgd_alpha"], a["pgd_steps"], device)

    def predict(name, X):
        if name == "rf":
            return rf_predict(models[name], X)
        return torch_predict(models[name], X, device)

    def record(name, attack, eps, X):
        pred, score = predict(name, X)
        rows.append({"model": name, "attack": attack, "eps": eps, **metrics(yte, pred, score)})

    for name in models:
        record(name, "clean", 0.0, Xte)

    attacks = (("fgsm", fgsm, {}),
               ("pgd", pgd, {"alpha": a["pgd_alpha"], "steps": a["pgd_steps"]}))

    # White-box attacks; RF ko NN ke against bane examples se evaluate kiya hai (transfer).
    for eps in ev["eps"]:
        for name, m in models.items():
            if name == "rf":
                continue
            for atk_name, fn, kw in attacks:
                record(name, atk_name, eps, batched_attack(fn, m, Xte, yte, device, eps=eps, **kw))
        if "rf" in models and "nn" in models:
            for atk_name, fn, kw in attacks:
                Xadv = batched_attack(fn, models["nn"], Xte, yte, device, eps=eps, **kw)
                record("rf", atk_name + "_transfer_from_nn", eps, Xadv)

    # Black-box: surrogate ADV_NN ko query kar ke train hota hai
    if "adv_nn" in models:
        target = lambda X: torch_predict(models["adv_nn"], X, device)[0]
        sur = build_surrogate(target, Xte, ev["blackbox_budget"], device, cfg["seed"])
        for eps in ev["eps"]:
            Xadv = batched_attack(pgd, sur, Xte, yte, device, eps=eps,
                                  alpha=a["pgd_alpha"], steps=a["pgd_steps"])
            for name in models:
                record(name, "blackbox", eps, Xadv)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()