import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

samples = ["BuToKJPsi", "PromptJPsi"]
colors  = {"BuToKJPsi": "C0", "PromptJPsi": "C3"}
var, xlabel = "el_pt", r"Probe $p_T$ [GeV]"

filters = [
    "passHltL1DoubleEGXer1p2dRMaxYFilter",
    "passHltDoubleEle6eta1p22EtFilter",
    "passHltDoubleEle6eta1p22PixelMatchFilter",
    "passHltDoubleEle6eta1p22PMS2Filter",
    "passHltDoubleEle6eta1p22OneOEMinusOneOPFilter",
    "passHltDoubleEle6eta1p22DetaFilter",
    "passHltDoubleEle6eta1p22DphiFilter",
    "passHltDoubleEle6eta1p22Chi2Filter",
    "passHltDoubleEle6eta1p22ValidHitsFilter",
    "passHltDoubleEle6eta1p22NLayerITFilter",
    "passHltDoubleEle6eta1p22PMmMax6MassFilter",
]

BASE = "."

def load(sample, filt):
    paths = glob.glob(f"{BASE}/{sample}/get_nd_tnp_histograms_*/{filt}_histos.pkl")
    assert len(paths) == 1, f"{sample} {filt}: found {paths}"
    with open(paths[0], "rb") as f:
        return pickle.load(f)


# Optional: narrow the mass window in post-processing (mll axis exists since cut_and_count=False)
MLL_WINDOW = None            # e.g. (2.8, 3.3); None = use the full axis

def to_1d(h, keep, mll_window=MLL_WINDOW):
    """Sum all other axes over in-range bins only (no flow)."""
    sl = {}
    for ax in h.axes:
        if ax.name == keep:
            continue
        if ax.name == "mll" and mll_window is not None:
            sl["mll"] = slice(complex(0, mll_window[0]), complex(0, mll_window[1]), sum)
        else:
            sl[ax.name] = slice(0, len(ax), sum)
    return h[sl]

def eff(num, den):
    from hist.intervals import clopper_pearson_interval
    n, d = num.values(), den.values()
    e = np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
    lo, hi = clopper_pearson_interval(n, d)
    return e, np.clip(e - lo, 0, None), np.clip(hi - e, 0, None)

def draw(ax, h_ref, e, elo, ehi, label, color):
    ax.errorbar(h_ref.axes[0].centers, e, xerr=h_ref.axes[0].widths / 2,
                yerr=[elo, ehi], fmt="o", ms=3, color=color, label=label)

data = {s: {f: tuple(load(s, f)[k] for k in ("passing", "failing")) for f in filters}
        for s in samples}

def short(f):
    return f.replace("passHltDoubleEle6eta1p22", "").replace("passHlt", "").replace("Filter", "")

def make_grid(mode, fname):
    ncols = 4
    nrows = int(np.ceil(len(filters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, f in enumerate(filters):
        ax = axes[i]
        for s in samples:
            p, fl = data[s][f]
            P = to_1d(p, var)
            if mode == "cumulative":          # passing_i / all probes
                den = to_1d(p + fl, var)
            else:                             # passing_i / passing_{i-1}
                if i == 0:
                    den = to_1d(p + fl, var)  # first step: relative to all probes
                else:
                    pprev, _ = data[s][filters[i - 1]]
                    den = to_1d(pprev, var)
            draw(ax, P, *eff(P, den), label=s, color=colors[s])
        ax.set_title(f"{i+1}: {short(f)}", fontsize=9)
        ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    for ax in axes[len(filters):]:
        ax.axis("off")
    for ax in axes[::ncols]: ax.set_ylabel("Efficiency")
    for ax in axes[-ncols:]: ax.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)

make_grid("cumulative", "compare_pt_cumulative_grid.png")
make_grid("relative",   "compare_pt_relative_grid.png")

# One standalone plot per filter, with a ratio panel
for i, f in enumerate(filters):
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    effs = {}
    for s in samples:
        p, fl = data[s][f]
        P, T = to_1d(p, var), to_1d(p + fl, var)
        e, elo, ehi = eff(P, T)
        effs[s] = e
        draw(ax, P, e, elo, ehi, s, colors[s])
    ratio = np.divide(effs["BuToKJPsi"], effs["PromptJPsi"],
                      out=np.full_like(effs["BuToKJPsi"], np.nan), where=effs["PromptJPsi"] > 0)
    rax.plot(P.axes[0].centers, ratio, "ko", ms=3)
    rax.axhline(1, color="gray", ls="--")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Efficiency (cumulative)"); ax.legend()
    ax.set_title(f"{i+1}: {short(f)}")
    rax.set_ylim(0.5, 1.5); rax.set_ylabel("Bu / Prompt"); rax.set_xlabel(xlabel)
    fig.savefig(f"compare_pt_{i+1:02d}_{short(f)}.png", dpi=150); plt.close(fig)
