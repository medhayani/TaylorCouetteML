"""Train HYDRA-MARL_PRO: 3-phase curriculum on Step F windows (bigger config).

Reuses code6's working 3-phase logic (MADT -> IQL -> MAPPO + diffusion policies)
but with the PRO config: 7 agents (incl. Trend, Asymmetry), bigger transformer
critic, more communication rounds.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import HydraWindowsDataset
from hydra_marl_pro.agents.validator import ValidatorAgent
from hydra_marl_pro.algorithms.iql import IQLCritic
from hydra_marl_pro.algorithms.madt import MultiAgentDecisionTransformer
from hydra_marl_pro.communication.gat import GATCommunication
from hydra_marl_pro.critic.multi_agent_transformer import MultiAgentTransformerCritic
from hydra_marl_pro.policies.diffusion_policy import DiffusionPolicy


# 7 agents: detector, shape, geometry, smoothness, trend, asymmetry, validator
def action_dims(window_T: int) -> list:
    return [3, 4, 5, window_T, 3, 4, 7]


def flatten_state(batch):
    B = batch["obs_seq"].size(0)
    return torch.cat([batch["obs_seq"].reshape(B, -1), batch["static_vec"],
                       batch["y_pred"]], dim=-1)


def heuristic_actions(batch, dims):
    res = batch["y_true"] - batch["y_pred"]
    B = res.size(0); device = res.device
    return [
        torch.zeros(B, dims[0], device=device),     # detector
        torch.zeros(B, dims[1], device=device),     # shape
        torch.zeros(B, dims[2], device=device),     # geometry
        res,                                          # smoothness
        torch.zeros(B, dims[4], device=device),     # trend
        torch.zeros(B, dims[5], device=device),     # asymmetry
        torch.ones(B, dims[6], device=device),      # validator (gates open)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_train", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("../data/runs_pro/hydra_pro"), type=Path)
    ap.add_argument("--madt_epochs", type=int, default=20)
    ap.add_argument("--iql_epochs", type=int, default=30)
    ap.add_argument("--mappo_steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = get_logger("hydra_pro.train")
    set_all_seeds(args.seed, deterministic=False)
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    h = full_cfg["hydra_pro"]
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = HydraWindowsDataset(args.windows_train.resolve())
    log.info(f"train windows: N={len(train_ds)}, "
             f"obs_seq={tuple(train_ds.obs_seq.shape)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                drop_last=True)
    T = train_ds.y_true.shape[1]
    state_dim = (train_ds.obs_seq.shape[1] * train_ds.obs_seq.shape[2]
                 + train_ds.static_vec.shape[1] + T)
    a_dims = action_dims(T)
    log.info(f"state_dim={state_dim}  action_dims={a_dims}")

    # PHASE 1 — MADT
    log.info("=" * 60); log.info(f"PHASE 1 — MADT ({args.madt_epochs} epochs)")
    madt = MultiAgentDecisionTransformer(state_dim=state_dim, action_dims=a_dims,
                                            d_model=h["critic"]["d_model"] // 2,
                                            num_layers=4, num_heads=8,
                                            context_length=4).to(device)
    optim = torch.optim.AdamW(madt.parameters(), lr=args.lr)
    log.info(f"MADT params={sum(p.numel() for p in madt.parameters())/1e6:.2f} M")
    for ep in range(1, args.madt_epochs + 1):
        agg, n = 0.0, 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            state = flatten_state(b); actions = heuristic_actions(b, a_dims)
            R = -((b["y_true"] - b["y_pred"]) ** 2).mean(dim=-1, keepdim=True).unsqueeze(1)
            s = state.unsqueeze(1)
            a_seq = [a.unsqueeze(1) for a in actions]
            preds = madt(R, s, a_seq)
            loss = sum(((p - a) ** 2).mean() for p, a in zip(preds, a_seq))
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(madt.parameters(), 1.0); optim.step()
            agg += float(loss); n += 1
        log.info(f"  MADT {ep:3d}/{args.madt_epochs}  loss={agg/max(n,1):.4f}")
    torch.save({"state_dict": madt.state_dict(), "state_dim": state_dim,
                 "action_dims": a_dims}, out / "madt.pt")

    # PHASE 2 — IQL
    log.info("=" * 60); log.info(f"PHASE 2 — IQL ({args.iql_epochs} epochs)")
    iql = IQLCritic(state_dim=state_dim, action_dims=a_dims, hidden=512).to(device)
    opt_v = torch.optim.AdamW(iql.V.parameters(), lr=args.lr)
    opt_q = torch.optim.AdamW(iql.Q.parameters(), lr=args.lr)
    for ep in range(1, args.iql_epochs + 1):
        lv, lq, n = 0.0, 0.0, 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            state = flatten_state(b); actions = heuristic_actions(b, a_dims)
            r = -((b["y_true"] - b["y_pred"]) ** 2).mean(dim=-1)
            done = torch.ones_like(r); next_state = state
            l_v = iql.value_loss(state, actions)
            opt_v.zero_grad(); l_v.backward()
            torch.nn.utils.clip_grad_norm_(iql.V.parameters(), 1.0); opt_v.step()
            l_q = iql.q_loss(state, actions, r, next_state, done)
            opt_q.zero_grad(); l_q.backward()
            torch.nn.utils.clip_grad_norm_(iql.Q.parameters(), 1.0); opt_q.step()
            iql.soft_update()
            lv += float(l_v); lq += float(l_q); n += 1
        log.info(f"  IQL {ep:3d}/{args.iql_epochs}  L_V={lv/max(n,1):.4f}  L_Q={lq/max(n,1):.4f}")
    torch.save({"state_dict": iql.state_dict(), "state_dim": state_dim,
                 "action_dims": a_dims}, out / "iql.pt")

    # PHASE 3 — MAPPO + diffusion policies + GAT + MAT
    log.info("=" * 60); log.info(f"PHASE 3 — MAPPO+diff ({args.mappo_steps} steps)")
    mat = MultiAgentTransformerCritic(state_dim=state_dim, action_dims=a_dims,
                                        d_model=h["critic"]["d_model"],
                                        num_layers=h["critic"]["num_layers"],
                                        num_heads=h["critic"]["num_heads"],
                                        dropout=h["critic"]["dropout"]).to(device)
    gat = GATCommunication(dim=state_dim,
                             num_rounds=h["communication"]["num_rounds"],
                             num_heads=1).to(device)
    msg_dim = state_dim
    policies = [
        DiffusionPolicy(action_dim=a_dims[i], state_dim=state_dim, msg_dim=msg_dim,
                          num_timesteps=h["diffusion_policy"]["num_timesteps"]).to(device)
        for i in range(6)                                # 6 corrector agents
    ]
    validator = ValidatorAgent(obs_dim=state_dim,
                                  action_dim=a_dims[6],
                                  hidden_dim=h["agents"]["validator"]["hidden_dim"]).to(device)
    log.info(f"MAT params={sum(p.numel() for p in mat.parameters())/1e6:.2f}M  "
             f"GAT params={sum(p.numel() for p in gat.parameters())/1e6:.2f}M  "
             f"policies x{len(policies)}")
    optim_pol = torch.optim.AdamW(
        [p for n in policies for p in n.parameters()] + list(validator.parameters()),
        lr=args.lr)
    optim_crit = torch.optim.AdamW(mat.parameters(), lr=args.lr)

    step = 0; iter_loader = iter(train_loader)
    while step < args.mappo_steps:
        try:
            b = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader); b = next(iter_loader)
        b = {k: v.to(device) for k, v in b.items()}
        state = flatten_state(b); actions = heuristic_actions(b, a_dims)
        h_init = state.unsqueeze(1).expand(-1, 7, -1).contiguous()
        h_post = gat(h_init)
        with torch.no_grad():
            adv = iql.advantage(state, actions).detach()
        diff_loss = 0.0
        for i in range(6):
            diff_loss = diff_loss + policies[i].loss(actions[i], state,
                                                       h_post[:, i, :], advantage=adv)
        with torch.no_grad():
            target_q = iql.Q(torch.cat([state, iql._flatten_actions(actions)], dim=-1))
        crit_loss = nn.functional.mse_loss(mat(state, actions), target_q.detach())
        loss = diff_loss + 0.5 * crit_loss
        optim_pol.zero_grad(); optim_crit.zero_grad(); loss.backward()
        optim_pol.step(); optim_crit.step()
        step += 1
        if step % 50 == 0 or step == 1:
            log.info(f"  step {step:4d}/{args.mappo_steps}  diff={float(diff_loss):.4f}  "
                     f"critic={float(crit_loss):.4f}  adv_mean={float(adv.mean()):.4f}")
    torch.save({"policies": [p.state_dict() for p in policies],
                 "validator": validator.state_dict(),
                 "mat": mat.state_dict(), "gat": gat.state_dict(),
                 "state_dim": state_dim, "action_dims": a_dims},
                out / "mappo_phase3.pt")
    log.info("HYDRA_PRO ALL PHASES DONE")


if __name__ == "__main__":
    main()
