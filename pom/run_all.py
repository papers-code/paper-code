#!/usr/bin/env python3
# Regenerates the manuscript's CSVs and figures. Run from the package directory.
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
np.random.seed(20260701)

N0 = 25
sigma_xi2 = 0.10
sigma_eps2 = 1.00
sigma_zeta2 = 0.50
lam = 1.0
sigma_u2 = 0.10

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def VL(beta, su=sigma_u2, sx=sigma_xi2):
    return beta**2*(su+sx) + (1-beta)**2*sigma_eps2 + sigma_zeta2

def VA(beta, N=N0, su=sigma_u2, sx=sigma_xi2):
    return N*N*beta**2*su + N*beta**2*sx + N*(1-beta)**2*sigma_eps2 + N*sigma_zeta2

# fig1
betas = np.linspace(0, 1, 101)
VLs = np.array([VL(b) for b in betas])
VAs = np.array([VA(b) for b in betas])
bL = sigma_eps2/(sigma_eps2+sigma_xi2+sigma_u2)
bA = sigma_eps2/(sigma_eps2+sigma_xi2+N0*sigma_u2)
write_csv(
    ROOT/"data_figure1_reversal_region.csv",
    ["beta", "local_norm", "aggregate_norm"],
    ([f"{b:.4f}", f"{vl:.6f}", f"{va:.6f}"]
     for b, vl, va in zip(betas, VLs/VLs.min(), VAs/VAs.min()))
)

plt.figure(figsize=(6.4, 4.2))
plt.plot(betas, VLs / VLs.min(), label="Local variance")
plt.plot(betas, VAs / VAs.min(), label="Aggregate variance")
plt.axvline(bA, linestyle="--", linewidth=1, label=r"$\beta_A^\star$")
plt.axvline(bL, linestyle=":", linewidth=1, label=r"$\beta_L^\star$")
plt.xlabel(r"AI reliance weight $\beta$")
plt.ylabel("Normalized variance")
plt.title("Local accuracy--aggregate volatility reversal")
plt.legend()
save_current_figure(ROOT/"figure1_reversal_region.pdf")

# fig2
Ns = np.arange(2, 101)
betaD = np.repeat(sigma_eps2/(sigma_eps2+sigma_xi2+sigma_u2), len(Ns))
betaS = np.array([sigma_eps2/(sigma_eps2+sigma_xi2+sigma_u2+lam*(n-1)*sigma_u2) for n in Ns])
write_csv(
    ROOT/"data_figure2_reliance_vs_network_size.csv",
    ["N", "decentralized", "system"],
    ([int(n), f"{bd:.6f}", f"{bs:.6f}"] for n, bd, bs in zip(Ns, betaD, betaS))
)

plt.figure(figsize=(6.4, 4.2))
plt.plot(Ns, betaD, label="Decentralized reliance")
plt.plot(Ns, betaS, label="System-optimal reliance")
plt.xlabel("Number of downstream units N")
plt.ylabel(r"Reliance weight $\beta$")
plt.title("Decentralized reliance vs. system optimum")
plt.legend()
save_current_figure(ROOT/"figure2_reliance_vs_network_size.pdf")

# fig3
N = 100
beta = 0.5
su = 0.05
groups = np.array([1,2,4,5,10,20,25,50])
peak = su*(np.ceil(N/groups)*beta)**2
family = su*(N*beta)**2/groups
write_csv(
    ROOT/"data_figure3_governance_exposure.csv",
    ["groups", "staggered_peak", "split_family_exposure"],
    ([int(g), f"{p:.6f}", f"{fam:.6f}"] for g, p, fam in zip(groups, peak, family))
)

plt.figure(figsize=(6.4, 4.2))
plt.plot(groups, peak, marker="o", label="Staggered update peak exposure")
plt.plot(groups, family, marker="s", label="Split-family common exposure")
plt.xlabel("Number of groups/families")
plt.ylabel("Common-error exposure")
plt.title("Diversification and staggering reduce common exposure")
plt.legend()
save_current_figure(ROOT/"figure3_governance_exposure.pdf")

rng = np.random.default_rng(20260701)
T = 800
Nret = 30
rho = 0.65
theta = np.zeros(T)
season = 0.4*np.sin(2*np.pi*np.arange(T)/52.0)
for t in range(1, T):
    theta[t] = rho*theta[t-1] + rng.normal(0, 1.0)
theta = theta + season
zeta = rng.normal(0, 0.7, size=(T, Nret))
demand = theta[:,None] + zeta

def run_scenario(name, beta_ai, sigma_u, sigma_xi, sigma_eps, families=1, capacity_k=12.0):
    eps = rng.normal(0, sigma_eps, size=(T, Nret))
    private = theta[:,None] + eps
    family_ids = np.arange(Nret) % families
    u_family = rng.normal(0, sigma_u, size=(T, families))
    xi = rng.normal(0, sigma_xi, size=(T, Nret))
    ai = theta[:,None] + u_family[:, family_ids] + xi
    orders = beta_ai*ai + (1-beta_ai)*private
    residual = orders - demand
    cov = np.cov(residual.T)
    off = cov[~np.eye(Nret, dtype=bool)]
    agg = residual.sum(axis=1)
    return {
        "scenario": name,
        "local_rmse": float(np.sqrt(np.mean(residual**2))),
        "avg_residual_cov": float(np.mean(off)),
        "aggregate_volatility": float(np.var(agg)),
        "capacity_tail_prob": float(np.mean(np.abs(agg) > capacity_k)),
        "total_cost": float(np.mean(np.sum(residual**2, axis=1) + 0.2*(agg**2) + 0.8*(np.maximum(agg-capacity_k, 0)**2 + np.maximum(-agg-capacity_k, 0)**2))),
    }

scenarios = [
    run_scenario("Private forecast", 0.0, 0.0, 0.0, 1.0, families=30),
    run_scenario("Common AI", 0.80, 0.45, 0.35, 1.0, families=1),
    run_scenario("Split AI families", 0.80, 0.45, 0.35, 1.0, families=5),
    run_scenario("Diverse AI", 0.75, 0.22, 0.48, 1.0, families=5),
    run_scenario("Covariance-priced AI", 0.42, 0.45, 0.35, 1.0, families=1),
]
with open(ROOT/"data_illustrative_experiment.csv", "w", newline="") as f:
    cols = list(scenarios[0].keys())
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for row in scenarios:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k,v in row.items()})
