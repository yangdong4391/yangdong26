#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 特异性检验 —— 使用与原稿一致的 ssGSEA 评分方法（gseapy），
        以「表达量 × 方差」分层匹配的随机基因集构建零分布。

设计要点
--------
1. 评分方法固定为 ssGSEA（gseapy, sample_norm_method="rank"），与原稿 Table 3/Fig 6 一致。
2. 随机基因集：与目标模块等长，且在与目标模块相同的
   「平均表达十分位 × 表达方差四分位」分层上按比例抽样（expression/variance-matched）。
3. gseapy 单次评分耗时长且内存占用大（100 个基因集约 5.7 GB），
   因此把 n_perm 个基因集分批（默认 batch=100）评分，批间释放内存。
4. 输出：观察 d、零分布 summary、经验 P（单侧/双侧）、z_emp，
   以及 Null d 数组与图示。

用法
----
python3 m1_ssgsea_null.py \
    --expr  GSE65682_gene_matrix_log2.csv \
    --meta  meta_65682.csv \
    --module signatures/death_promoting_module.txt \
    --group_col group --sepsis sepsis --control healthy \
    --n_perm 5000 --batch 100 --seed 2026 --outdir M1_ssgsea
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

try:
    import gseapy as gp
except ImportError:  # pragma: no cover
    raise SystemExit("需要 gseapy：python3 -m pip install gseapy")


# ----------------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------------
def cohens_d(x, y) -> float:
    """合并标准差 Cohen d（x - y）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if s == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / s)


def build_strata(expr: pd.DataFrame, n_mean_bins: int = 10, n_sd_bins: int = 4) -> pd.Series:
    """按 (平均表达十分位 × 表达方差四分位) 给基因分层。"""
    mean_expr = expr.mean(axis=1)
    sd_expr = expr.std(axis=1, ddof=1)
    b_mean = pd.qcut(mean_expr.rank(method="first"), n_mean_bins, labels=False)
    b_sd = pd.qcut(sd_expr.rank(method="first"), n_sd_bins, labels=False)
    return (b_mean.astype(int).astype(str) + "_" + b_sd.astype(int).astype(str))


def expression_matched_draws(expr, module_genes, n_perm, seed=2026):
    """抽取 n_perm 个随机基因集：与模块等长，且分层构成完全一致。"""
    present = [g for g in module_genes if g in expr.index]
    strata = build_strata(expr)
    target = strata.loc[present].value_counts()

    pool = {}
    for k in target.index:
        idx = strata.index[strata == k].to_numpy()
        if len(idx) < target[k]:
            raise ValueError(f"分层 {k} 候选基因不足（需 {target[k]}，池内 {len(idx)}）")
        pool[k] = idx

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        genes = []
        for k, cnt in target.items():
            genes.extend(rng.choice(pool[k], size=int(cnt), replace=False).tolist())
        draws.append(genes)
    return draws


# ----------------------------------------------------------------------------
# ssGSEA 评分（gseapy；分批以控制内存）
# ----------------------------------------------------------------------------
def ssgsea_scores(expr: pd.DataFrame, gene_sets: dict, threads: int = 4,
                  batch: int = 100, verbose: bool = True) -> pd.DataFrame:
    """
    对多个基因集逐一（分批）计算 ssGSEA 得分矩阵。
    返回 DataFrame：index = 样本，columns = 基因集名。
    """
    cols = []
    keys = list(gene_sets.keys())
    for i in range(0, len(keys), batch):
        chunk_keys = keys[i:i + batch]
        chunk = {k: gene_sets[k] for k in chunk_keys}
        res = gp.ssgsea(data=expr, gene_sets=chunk, outdir=None,
                        sample_norm_method="rank", correl_norm_type="rank",
                        min_size=5, max_size=5000, permutation_num=0,
                        threads=threads, no_plot=True)
        r2 = res.res2d.rename(columns={"Name": "sample"})
        piv = (r2.pivot(index="sample", columns="Term", values="ES")
                 .reindex(expr.columns))
        cols.append(piv)
        if verbose:
            print(f"    ssGSEA 批次 {i // batch + 1}/{(len(keys) + batch - 1) // batch}"
                  f" 完成（{len(chunk_keys)} 个基因集）", flush=True)
        del res, r2
    return pd.concat(cols, axis=1)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--sepsis", default="sepsis")
    ap.add_argument("--control", default="healthy")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default="M1_ssgsea")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    print("=" * 78)
    print("M1 特异性检验（ssGSEA 评分 + 表达/方差匹配置换零模型）")
    print("=" * 78)

    expr = pd.read_csv(a.expr, index_col=0).astype(float)
    meta = pd.read_csv(a.meta, index_col=0)
    meta = meta.loc[[c for c in expr.columns if c in meta.index]]
    expr = expr[meta.index]
    print(f"表达矩阵: {expr.shape[0]} 基因 × {expr.shape[1]} 样本")

    module = [l.strip() for l in open(a.module) if l.strip()]
    present = [g for g in module if g in expr.index]
    print(f"模块基因: {len(module)} 定义 → {len(present)} 可测（缺失 {len(module) - len(present)}）")
    if len(present) < 5:
        raise SystemExit("可测基因过少")

    grp = meta[a.group_col].astype(str)
    is_sep = (grp == a.sepsis).values
    is_ctl = (grp == a.control).values
    print(f"分组: {a.sepsis} n={is_sep.sum()}, {a.control} n={is_ctl.sum()}")
    keep = is_sep | is_ctl
    expr_k = expr.loc[:, keep]
    is_sep_k = is_sep[keep]

    # ---- 观察值 ----
    print("\n[1/4] 计算观察模块的 ssGSEA 得分 ...")
    obs = ssgsea_scores(expr_k, {"MODULE": present}, threads=a.threads, batch=1)
    s_obs = obs["MODULE"].values
    d_obs = cohens_d(s_obs[is_sep_k], s_obs[~is_sep_k])
    t_obs, p_obs = stats.ttest_ind(s_obs[is_sep_k], s_obs[~is_sep_k], equal_var=False)
    print(f"    观察 Cohen d = {d_obs:+.4f}  (Welch P = {p_obs:.3e})")

    # ---- 零分布 ----
    print(f"\n[2/4] 生成 {a.n_perm} 个表达/方差匹配随机基因集 ...")
    draws = expression_matched_draws(expr_k, present, a.n_perm, seed=a.seed)
    sets = {f"PERM{i:05d}": g for i, g in enumerate(draws)}
    print(f"\n[3/4] 分批 ssGSEA 评分 ...")
    t0 = time.time()
    sc = ssgsea_scores(expr_k, sets, threads=a.threads, batch=a.batch)
    print(f"    完成，用时 {time.time() - t0:.1f}s")

    d_null = np.array([cohens_d(sc[c].values[is_sep_k], sc[c].values[~is_sep_k])
                       for c in sc.columns])
    np.save(os.path.join(a.outdir, "null_d_module.npy"), d_null)

    p_emp_one = (np.sum(d_null >= d_obs) + 1) / (a.n_perm + 1)
    p_emp_two = (np.sum(np.abs(d_null) >= abs(d_obs)) + 1) / (a.n_perm + 1)
    z_emp = (d_obs - d_null.mean()) / d_null.std(ddof=1)

    print("\n[4/4] 结果")
    print("-" * 78)
    print(f"  模块可测基因数      : {len(present)}")
    print(f"  观察 Cohen d        : {d_obs:+.4f}  (Welch P = {p_obs:.3e})")
    print(f"  零分布              : mean {d_null.mean():+.4f} ± SD {d_null.std(ddof=1):.4f}")
    print(f"                        95% 分位 {np.quantile(d_null, .95):+.4f}"
          f"；99% 分位 {np.quantile(d_null, .99):+.4f}")
    print(f"  经验 P（单侧，d≥obs）: {p_emp_one:.4g}")
    print(f"  经验 P（双侧）       : {p_emp_two:.4g}")
    print(f"  标准化效应 z         : {z_emp:+.3f}")
    print("-" * 78)

    out = dict(
        score_method="ssGSEA (gseapy, sample_norm_method=rank)",
        n_perm=a.n_perm, seed=a.seed, batch=a.batch,
        n_present=len(present), n_missing=len(module) - len(present),
        d_obs=d_obs, p_obs=float(p_obs),
        null_mean=float(d_null.mean()), null_sd=float(d_null.std(ddof=1)),
        null_q95=float(np.quantile(d_null, .95)), null_q99=float(np.quantile(d_null, .99)),
        p_emp_one=float(p_emp_one), p_emp_two=float(p_emp_two), z_emp=float(z_emp),
        n_sepsis=int(is_sep_k.sum()), n_control=int((~is_sep_k).sum()),
    )
    with open(os.path.join(a.outdir, "M1_ssgsea_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- 图示 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.hist(d_null, bins=60, color="#c9ced6", edgecolor="white", linewidth=0.4)
        ax.axvline(d_obs, color="#c0392b", lw=2,
                   label=f"Observed d = {d_obs:.2f}")
        ax.axvline(np.quantile(d_null, .95), color="#2c3e50", ls="--", lw=1.2,
                   label=f"95th percentile = {np.quantile(d_null, .95):.2f}")
        ax.set_xlabel("Cohen d (sepsis vs healthy), expression-matched random gene sets")
        ax.set_ylabel("Frequency")
        ax.set_title(f"ssGSEA permutation null (n = {a.n_perm}); empirical P = {p_emp_one:.4g}")
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.outdir, f"Fig_M1_ssgsea_null.{ext}"), dpi=200)
        print(f"  图已保存: {a.outdir}/Fig_M1_ssgsea_null.png / .pdf")
    except Exception as e:  # pragma: no cover
        print(f"  [警告] 绘图失败: {e}")

    print(f"\n结果已保存到 {a.outdir}/")


if __name__ == "__main__":
    main()
