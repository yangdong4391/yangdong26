#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M4-xCell：用官方 xCell（R 包，67 种细胞类型）做细胞组成校正
================================================================
与 M4 原脚本（PanglaoDB 标志基因）的区别：
  --cells xcell_results.csv  即用官方 xCell 输出替换本地反卷积

两种变量选择方案（都跑，便于对照）：
  (A) 自动：非零方差 → VIF 剪枝（阈值可调）
  (B) 核心集：稿件原四个方向对应的 xCell 类别
      （Neutrophils, T-cells 合计, Monocytes, B-cells 合计）
"""
import argparse, os, json, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings("ignore")

GROUP = "group"


def zscore_rows(df):
    mu = df.mean(axis=1); sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def signature_score(expr, genes, method="ssgsea"):
    present = [g for g in genes if g in expr.index]
    if len(present) < 5:
        raise ValueError(f"签名匹配基因过少（{len(present)}）")
    if method == "zmean":
        return zscore_rows(expr.loc[present]).mean(axis=0), len(present)
    import gseapy as gp
    res = gp.ssgsea(data=expr, gene_sets={"SIG": present}, outdir=None,
                    sample_norm_method="rank", correl_norm_type="rank",
                    min_size=5, max_size=5000, permutation_num=0,
                    threads=4, no_plot=True)
    r2 = res.res2d.rename(columns={"Name": "sample"})
    return r2.set_index("sample")["ES"].reindex(expr.columns), len(present)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1); r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(r, 0, 1); return out


def prune_vif(X, thresh=10.0, verbose=False):
    cols = list(X.columns); dropped = []
    for _ in range(300):
        if len(cols) < 2:
            break
        vifs = []
        for c in cols:
            others = [k for k in cols if k != c]
            if not others:
                vifs.append((c, 1.0)); continue
            r2 = sm.OLS(X[c], sm.add_constant(X[others])).fit().rsquared
            vifs.append((c, 1.0 / max(1e-12, 1.0 - r2)))
        w, v = max(vifs, key=lambda t: t[1])
        if v <= thresh:
            break
        cols.remove(w); dropped.append((w, float(v)))
    if verbose:
        print(f"    VIF 剪枝：{len(X.columns)} → {len(cols)}（剔除 {len(dropped)}）")
    return cols, dropped


def run(expr, meta, signatures, cells, covars, sep_lbl, ctl_lbl,
        score_method, vif_thresh, core_cols=None):
    sep_ids = meta.index[meta[GROUP].astype(str) == sep_lbl]
    ctl_ids = meta.index[meta[GROUP].astype(str) == ctl_lbl]
    keep = [c for c in expr.columns if c in set(sep_ids) | set(ctl_ids)]
    expr_k = expr[keep]

    base = pd.DataFrame(index=expr_k.columns)
    base[GROUP] = np.where(expr_k.columns.isin(sep_ids), 1, 0)
    for c in covars:
        if c in meta.columns:
            base[c] = meta.loc[expr_k.columns, c].values
    used_covars = [c for c in covars if c in base.columns]

    # 细胞变量
    cell_sets = {}
    if cells is not None:
        ca = cells.reindex(expr_k.columns)
        ca = ca.loc[:, ca.std(ddof=1) > 1e-8].dropna(axis=1)
        kept_auto, dropped_auto = prune_vif(ca, vif_thresh)
        cell_sets["M2_xCell_autoVIF"] = (ca[kept_auto], kept_auto, dropped_auto, ca.columns, "auto")
        if core_cols:
            core = [c for c in core_cols if c in ca.columns]
            cell_sets["M2_xCell_core"] = (ca[core], core, [], ca.columns, "core")

    rows = []
    for sig_name, genes in signatures.items():
        score, n_present = signature_score(expr_k, genes, score_method)
        df = base.copy()
        df["_score"] = score.reindex(df.index)
        models = {"M0_unadjusted": [GROUP] + used_covars}
        for mname, (mat, used, dropped, allcols, kind) in cell_sets.items():
            models[mname] = [GROUP] + used_covars + used
        for mname, terms in models.items():
            if mname in cell_sets:
                mat = cell_sets[mname][0]
                d = df.copy()
                for c in mat.columns:
                    d[c] = mat[c].values
                d = d.dropna()
            else:
                d = df.dropna()
            Xd = sm.add_constant(d[terms], has_constant="add")
            try:
                fit = sm.OLS(d["_score"], Xd).fit(cov_type="HC3")
            except Exception as e:
                print(f"    [跳过] {sig_name}/{mname}: {e}"); continue
            rows.append(dict(signature=sig_name, n_genes=n_present, model=mname,
                             n=len(d), n_covariates=len(terms) - 1,
                             beta=float(fit.params[GROUP]), se=float(fit.bse[GROUP]),
                             p=float(fit.pvalues[GROUP]), r2=float(fit.rsquared_adj),
                             cells_used=(";".join(terms[1 + len(used_covars):]))))
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_fdr"] = res.groupby("model")["p"].transform(lambda s: bh(s.values))
    info = dict(n_sepsis=int(base[GROUP].sum()), n_control=int((1 - base[GROUP]).sum()),
                covars=used_covars,
                auto_cells=cell_sets.get("M2_xCell_autoVIF", (None, [], [], [], ""))[1],
                auto_dropped=[d[0] for d in cell_sets.get("M2_xCell_autoVIF", (None, [], [], [], ""))[2]],
                core_cells=cell_sets.get("M2_xCell_core", (None, [], [], [], ""))[1],
                score_method=score_method, vif_thresh=vif_thresh)
    return res, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--sigdir", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--covars", default="age,sex")
    ap.add_argument("--core_cells", default="")
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--sepsis", default="sepsis")
    ap.add_argument("--control", default="healthy")
    ap.add_argument("--score_method", default="ssgsea", choices=["ssgsea", "zmean"])
    ap.add_argument("--vif", type=float, default=10.0)
    ap.add_argument("--outdir", default="M4_xCell_out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    print("=" * 80)
    print("M4-xCell：官方 xCell 反卷积校正")
    print("=" * 80)
    expr = pd.read_csv(a.expr, index_col=0).astype(float)
    meta = pd.read_csv(a.meta, index_col=0)
    sigs = {os.path.splitext(os.path.basename(f))[0]:
            [l.strip() for l in open(f) if l.strip()]
            for f in sorted(glob.glob(os.path.join(a.sigdir, "*.txt")))}
    cells = pd.read_csv(a.cells, index_col=0)
    core = [c.strip() for c in a.core_cells.split(",") if c.strip()]
    print(f"表达 {expr.shape}；签名 {len(sigs)}；xCell {cells.shape}；VIF 阈值 {a.vif}")

    res, info = run(expr, meta, sigs, cells, [c.strip() for c in a.covars.split(",") if c.strip()],
                    a.sepsis, a.control, a.score_method, a.vif, core)
    res.to_csv(os.path.join(a.outdir, "xcell_adjustment_results.csv"), index=False)
    json.dump(info, open(os.path.join(a.outdir, "xcell_adjustment_summary.json"), "w"),
              indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("结果（β = 脓毒症 vs 健康；Z 尺度）")
    print("=" * 80)
    show = res.copy()
    for c in ["beta", "se", "r2"]:
        show[c] = show[c].round(4)
    show["p"] = show["p"].map(lambda v: f"{v:.3e}")
    show["p_fdr"] = show["p_fdr"].map(lambda v: f"{v:.3e}")
    print(show[["signature", "model", "n_genes", "n_covariates", "beta", "se", "r2", "p", "p_fdr"]]
          .to_string(index=False))

    print(f"\n自动 VIF 保留 {len(info['auto_cells'])} 个细胞类型；核心集 {info['core_cells']}")
    print(f"输出：{a.outdir}/")


if __name__ == "__main__":
    main()
