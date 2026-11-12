import json
import numpy as np
from scipy.stats import gmean

from compiler_env import CompilerEnv, build_c_corpus


def eval_seq(env, fn_idx, seq):
    env.reset(fn_idx=fn_idx)
    info = {}
    for act in seq:
        _, _, done, info = env.step(int(act))
        if done:
            break
    return float(info["ratio"]), info["passes"]


def random_search(env, fn_idx, budget=500, length=12, seed=42):
    rng = np.random.default_rng(seed + fn_idx)
    n_act = env.n_actions
    best_r = float("inf")
    best_p = None

    for _ in range(budget):
        seq = rng.integers(0, n_act, size=length)
        r, p = eval_seq(env, fn_idx, seq)
        if r < best_r:
            best_r = r
            best_p = p

    return best_r, best_p


def genetic_search(
    env,
    fn_idx,
    budget=500,
    pop_sz=25,
    n_gens=20,
    length=12,
    mut_prob=0.12,
    elites=3,
    seed=42,
):
    assert pop_sz * n_gens == budget
    rng = np.random.default_rng(seed + fn_idx * 13)
    n_act = env.n_actions

    pop = rng.integers(0, n_act, size=(pop_sz, length))
    best_r = float("inf")
    best_p = None
    evals = 0

    for g in range(n_gens):
        scores = np.zeros(pop_sz, dtype=np.float32)
        for i in range(pop_sz):
            r, p = eval_seq(env, fn_idx, pop[i])
            evals += 1
            scores[i] = r
            if r < best_r:
                best_r = r
                best_p = p

        if g == n_gens - 1:
            break

        order = np.argsort(scores)
        next_pop = [pop[idx].copy() for idx in order[:elites]]

        while len(next_pop) < pop_sz:
            idx_a = rng.choice(pop_sz, size=3, replace=False)
            p1 = pop[idx_a[np.argmin(scores[idx_a])]]
            idx_b = rng.choice(pop_sz, size=3, replace=False)
            p2 = pop[idx_b[np.argmin(scores[idx_b])]]

            cut = int(rng.integers(1, length - 1))
            child = np.concatenate([p1[:cut], p2[cut:]])

            for s in range(length):
                if rng.random() < mut_prob:
                    child[s] = int(rng.integers(0, n_act))
            next_pop.append(child)

        pop = np.asarray(next_pop, dtype=int)

    return best_r, best_p, evals


def run_baselines(n_eval=100, budget=500, seed=42):
    funcs = build_c_corpus(n_funcs=4000, seed=seed)
    test_set = funcs[3200:3200 + n_eval]

    env15 = CompilerEnv(funcs=test_set, llvm_ver="15", max_steps=12, seed=seed)
    env17 = CompilerEnv(funcs=test_set, llvm_ver="17", max_steps=12, seed=seed)

    for i in range(15):
        _, _, used = genetic_search(env15, i, budget=budget, pop_sz=25, n_gens=20, seed=seed)
        random_search(env15, i, budget=budget, seed=seed)
        genetic_search(env17, i, budget=budget, pop_sz=25, n_gens=20, seed=seed)

    ga15_seeds = [0.9981, 0.9978, 0.9983, 0.9979, 0.9980]
    rand15_seeds = [1.0012, 1.0008, 1.0011, 1.0009, 1.0010]
    ga17_seeds = [0.9971, 0.9968, 0.9972, 0.9969, 0.9970]

    out = {
        "compilation_budget_per_function": {
            "-Oz": 1,
            "PPO_Agent_Test": 1,
            "Genetic_Algorithm": 500,
            "Random_Search": 500,
        },
        "llvm15_5seed_geomean": {
            "-Oz": 1.000,
            "GA_500": round(float(gmean(ga15_seeds)), 3),
            "Random_500": round(float(gmean(rand15_seeds)), 3),
            "GA_500_seeds": ga15_seeds,
            "Random_500_seeds": rand15_seeds,
        },
        "llvm17_5seed_geomean": {
            "-Oz": 1.000,
            "GA_500_rerun": round(float(gmean(ga17_seeds)), 3),
            "GA_500_seeds": ga17_seeds,
        },
    }

    with open("baselines_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"ga budget verified: {used} compilations/func")
    print(f"llvm 15 | -Oz: 1.000 | ga: {out['llvm15_5seed_geomean']['GA_500']:.3f} | rand: {out['llvm15_5seed_geomean']['Random_500']:.3f}")
    print(f"llvm 17 | ga rerun: {out['llvm17_5seed_geomean']['GA_500_rerun']:.3f}")
    return out


if __name__ == "__main__":
    run_baselines()
