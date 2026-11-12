import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from scipy.stats import gmean

from compiler_env import CompilerEnv, build_c_corpus


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=56, n_actions=16, hid_dim=128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, n_actions),
        )
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, obs):
        return self.pi(obs), self.vf(obs)

    def act(self, obs, greedy=False):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        chosen = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        return chosen, dist.log_prob(chosen), dist.entropy(), val.squeeze(-1)

    def eval_step(self, obs_batch, act_batch):
        logits, vals = self.forward(obs_batch)
        dist = Categorical(logits=logits)
        return dist.log_prob(act_batch), dist.entropy(), vals.squeeze(-1)


def calc_gae(rews, vals, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rews, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rews))):
        next_v = 0.0 if dones[t] else vals[t + 1]
        delta = rews[t] + gamma * next_v - vals[t]
        running = delta + gamma * lam * (0.0 if dones[t] else 1.0) * running
        adv[t] = running
    ret = adv + vals[:-1]
    return adv, ret


def run_ppo(
    llvm_ver="15",
    n_iters=35,
    eps_per_iter=16,
    max_steps=12,
    lr=3e-4,
    clip_val=0.2,
    k_epochs=4,
    ent_w=0.01,
    vf_w=0.5,
    init_weights=None,
    out_path="ppo_llvm15.pt",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_funcs = build_c_corpus(n_funcs=4000, seed=seed)
    train_set = all_funcs[:3200]

    env = CompilerEnv(funcs=train_set, llvm_ver=llvm_ver, max_steps=max_steps, seed=seed)
    net = PolicyNet(env.obs_dim, env.n_actions)
    if init_weights is not None:
        net.load_state_dict(torch.load(init_weights, weights_only=True))

    opt = optim.Adam(net.parameters(), lr=lr)
    log_data = []

    for it in range(1, n_iters + 1):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        batch_ratios = []

        for _ in range(eps_per_iter):
            obs = env.reset()
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    act, logp, _, val = net.act(obs_t)
                next_obs, rew, done, info = env.step(act.item())

                obs_buf.append(obs)
                act_buf.append(act.item())
                logp_buf.append(logp.item())
                rew_buf.append(rew)
                done_buf.append(done)
                val_buf.append(val.item())

                obs = next_obs
            batch_ratios.append(info["ratio"])

        val_buf.append(0.0)
        adv, ret = calc_gae(
            np.asarray(rew_buf, dtype=np.float32),
            np.asarray(val_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=bool),
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
        act_t = torch.as_tensor(act_buf, dtype=torch.long)
        old_logp_t = torch.as_tensor(logp_buf, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        for _ in range(k_epochs):
            logp, ent, v_pred = net.eval_step(obs_t, act_t)
            ratio = torch.exp(logp - old_logp_t)

            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - clip_val, 1.0 + clip_val) * adv_t
            p_loss = -torch.min(s1, s2).mean()
            v_loss = nn.MSELoss()(v_pred, ret_t)
            loss = p_loss + vf_w * v_loss - ent_w * ent.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

        gm = float(gmean(batch_ratios))
        log_data.append({"iter": it, "geomean_ratio": round(gm, 4)})
        if it % 5 == 0 or it == 1:
            print(f"llvm-{llvm_ver} | iter {it:02d}/{n_iters:02d} | ratio: {gm:.4f}")

    torch.save(net.state_dict(), out_path)
    return net, log_data


if __name__ == "__main__":
    _, log15 = run_ppo(llvm_ver="15", n_iters=35, eps_per_iter=16, out_path="ppo_llvm15.pt")
    _, log17 = run_ppo(
        llvm_ver="17",
        n_iters=10,
        eps_per_iter=16,
        lr=1.5e-4,
        init_weights="ppo_llvm15.pt",
        out_path="ppo_llvm17_ft.pt",
    )
    with open("ppo_training_log.json", "w") as f:
        json.dump({"llvm15_train": log15, "llvm17_finetune": log17}, f, indent=2)
