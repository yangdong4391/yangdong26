#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
M4 细胞组成校正：用正式反卷积（xCell / CIBERSORTx）替换 4 个标志物代理评分
================================================================================
目的：回答脓毒症转录组审稿人几乎必问的问题
  "Is the elevation of the module simply a consequence of the markedly altered
   leucocyte composition of septic blood?"

做法：对同一批签名，比较三种校正模型下的疾病效应
  M0  未校正               score ~ group
  M1  旧代理评分校正        score ~ group + age + sex + 4 个标志物代理
  M2  反卷积校正（推荐）     score ~ group + age + sex + xCell/CIBERSORTx 细胞组分

若 M2 下疾病项仍显著 → 头条结论"独立于白细胞组成"成立，可直接写进稿件
若 M2 下效应大幅衰减 → 需改写结论，并报告衰减幅度（这本身也是有价值的发现）

输入：
  --expr      基因 × 样本 表达矩阵 CSV
  --meta      样本注释 CSV（group / age / sex / 可选旧代理评分列）
  --sigdir    签名目录（每个 .txt 一个签名，文件名为签名名，每行一个基因）
  --cells     反卷积结果 CSV（样本 × 细胞类型，来自 xCell 或 CIBERSORTx）
  --proxies   旧代理评分在 meta 中的列名（逗号分隔，可选）
  --covars    其他协变量（如 age,sex）

输出：
  cell_adjustment_results.csv   三种模型下的 β / SE / P / FDR
  cell_adjustment_summary.json
  终端打印可直接粘贴进稿件的英文结果句

用法：
  python3 adjust_cell_composition.py \
      --expr GSE65682_gene_matrix.csv \
      --meta GSE65682_meta.csv \
      --sigdir signatures/ \
      --cells xcell_results.csv \
      --proxies neutrophil_proxy,tcell_proxy,monocyte_proxy,bcell_proxy \
      --group_col group --sepsis sepsis --control healthy \
      --covars age,sex --outdir M4_out
================================================================================
"""
import argparse
import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

GROUP_TERM = "group"


# ============================================================================
def zscore_rows(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def signature_score(expr, genes, method="ssgsea"):
    """
    计算签名得分。
      ssgsea: gseapy ssGSEA（与原稿 Table 3 / Fig 6 一致，默认）
      zmean : 基因 z-score 后取均值（稳健备选）
    """
    present = [g for g in genes if g in expr.index]
    if len(present) < 5:
        raise ValueError(f"签名匹配基因过少（{len(present)}），请检查基因符号")
    if method == "zmean":
        return zscore_rows(expr.loc[present]).mean(axis=0), len(present)
    import gseapy as gp
    res = gp.ssgsea(data=expr, gene_sets={"SIG": present}, outdir=None,
                    sample_norm_method="rank", correl_norm_type="rank",
                    min_size=5, max_size=5000, permutation_num=0,
                    threads=4, no_plot=True)
    r2 = res.res2d.rename(columns={"Name": "sample"})
    score = r2.set_index("sample")["ES"].reindex(expr.columns)
    return score, len(present)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def prune_by_vif(X, thresh=20.0, max_iter=60):
    """逐步剔除 VIF 过高的变量，避免细胞类型之间共线导致系数不稳"""
    cols = list(X.columns)
    for _ in range(max_iter):
        if len(cols) < 2:
            break
        Xi = sm.add_constant(X[cols], has_constant="add")
        vifs = []
        for c in cols:
            others = [k for k in cols if k != c]
            if not others:
                vifs.append((c, 1.0))
                continue
            r2 = sm.OLS(X[c], sm.add_constant(X[others], has_constant="add")).fit().rsquared
            vifs.append((c, 1.0 / max(1e-12, 1.0 - r2)))
        worst, v = max(vifs, key=lambda t: t[1])
        if v <= thresh:
            break
        cols.remove(worst)
    return cols


def fit_models(expr, meta, signatures, cells, proxies, covars,
               group_col, sepsis_lbl, control_lbl, score_method="ssgsea"):
    sepsis_ids = meta.index[meta[group_col].astype(str) == sepsis_lbl]
    control_ids = meta.index[meta[group_col].astype(str) == control_lbl]
    keep = [c for c in expr.columns if c in set(sepsis_ids) | set(control_ids)]
    expr_k = expr[keep]

    df = pd.DataFrame(index=expr_k.columns)
    df[GROUP_TERM] = np.where(expr_k.columns.isin(sepsis_ids), 1, 0)

    for c in covars:
        if c in meta.columns:
            df[c] = meta.loc[expr_k.columns, c].values
    used_covars = [c for c in covars if c in df.columns]

    # 旧代理评分
    used_proxies = []
    for c in proxies:
        if c in meta.columns:
            df[c] = meta.loc[expr_k.columns, c].values
            used_proxies.append(c)

    # 反卷积细胞组分：对齐样本、剔除近零方差、VIF 剪枝
    if cells is not None:
        cells_aligned = cells.reindex(expr_k.columns)
        cells_aligned = cells_aligned.loc[:, cells_aligned.std(ddof=1) > 1e-8]
        cells_aligned = cells_aligned.dropna(axis=1)
        kept = prune_by_vif(cells_aligned)
        dropped = [c for c in cells_aligned.columns if c not in kept]
        cells_final = cells_aligned[kept]
        for c in cells_final.columns:
            df[c] = cells_final[c].values
    else:
        cells_final, dropped = None, []

    df = df.dropna()
    n = len(df)

    models = {
        "M0_unadjusted": [GROUP_TERM],
        "M1_proxy_scores": [GROUP_TERM] + used_covars + used_proxies,
        "M2_deconvolution": ([GROUP_TERM] + used_covars + list(cells_final.columns)
                             if cells_final is not None else None),
    }

    rows = []
    for sig_name, genes in signatures.items():
        score, n_present = signature_score(expr_k, genes, score_method)
        df["_score"] = score.reindex(df.index)

        for model_name, terms in models.items():
            if terms is None or len(terms) < 1:
                continue
            X = sm.add_constant(df[terms], has_constant="add")
            try:
                fit = sm.OLS(df["_score"], X).fit(cov_type="HC3")
            except Exception as e:
                print(f"    [跳过] {sig_name} / {model_name}: {e}")
                continue
            rows.append(dict(
                signature=sig_name, n_genes=n_present, model=model_name,
                n_samples=n,
                beta=float(fit.params[GROUP_TERM]),
                se=float(fit.bse[GROUP_TERM]),
                t=float(fit.tvalues[GROUP_TERM]),
                p=float(fit.pvalues[GROUP_TERM]),
                adj_r2=float(fit.rsquared_adj),
                n_covariates=len(terms) - 1,
            ))

    res = pd.DataFrame(rows)
    # 各模型内部对多个签名做 BH 校正
    if not res.empty:
        res["p_fdr"] = res.groupby("model")["p"].transform(lambda s: bh_adjust(s.values))

    return res, dict(n=n, n_sepsis=int(df[GROUP_TERM].sum()),
                     n_control=int((1 - df[GROUP_TERM]).sum()),
                     cells_used=list(cells_final.columns) if cells_final is not None else [],
                     cells_dropped=dropped,
                     covars=used_covars, proxies=used_proxies)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="M4 细胞组成校正")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--sigdir", required=True, help="签名目录，每个 .txt 一个签名")
    ap.add_argument("--cells", default=None, help="反卷积结果 CSV（样本×细胞类型）")
    ap.add_argument("--proxies", default="", help="旧代理评分列名，逗号分隔")
    ap.add_argument("--covars", default="", help="协变量列名，逗号分隔")
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--sepsis", default="sepsis")
    ap.add_argument("--control", default="healthy")
    ap.add_argument("--score_method", default="ssgsea", choices=["ssgsea", "zmean"],
                    help="签名评分方法：ssgsea（与原稿一致，默认）或 zmean")
    ap.add_argument("--outdir", default="M4_out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    covars = [c.strip() for c in a.covars.split(",") if c.strip()]
    proxies = [c.strip() for c in a.proxies.split(",") if c.strip()]

    print("[1/4] 读取数据")
    expr = pd.read_csv(a.expr, index_col=0)
    meta = pd.read_csv(a.meta, index_col=0)
    signatures = {}
    for f in sorted(glob.glob(os.path.join(a.sigdir, "*.txt"))):
        signatures[os.path.splitext(os.path.basename(f))[0]] = \
            [l.strip() for l in open(f) if l.strip()]
    print(f"    表达矩阵 {expr.shape}；签名 {len(signatures)} 个：{list(signatures)}")

    cells = None
    if a.cells and os.path.exists(a.cells):
        cells = pd.read_csv(a.cells, index_col=0)
        print(f"    反卷积矩阵 {cells.shape}（样本 × 细胞类型）")
    else:
        print("    [警告] 未提供反卷积矩阵，M2_deconvolution 模型将跳过")

    print("[2/4] 拟合三种校正模型（HC3 稳健标准误）")
    res, info = fit_models(expr, meta, signatures, cells, proxies, covars,
                           a.group_col, a.sepsis, a.control, a.score_method)
    info["score_method"] = a.score_method

    print("[3/4] 保存结果")
    res.to_csv(os.path.join(a.outdir, "cell_adjustment_results.csv"), index=False)
    with open(os.path.join(a.outdir, "cell_adjustment_summary.json"), "w") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("M4 结果：三种校正模型下的疾病效应（β = 脓毒症 vs 健康）")
    print("=" * 78)
    show = res.copy()
    show["beta"] = show["beta"].round(4)
    show["se"] = show["se"].round(4)
    show["p"] = show["p"].map(lambda v: f"{v:.3e}")
    show["p_fdr"] = show["p_fdr"].map(lambda v: f"{v:.3e}")
    print(show[["signature", "model", "n_genes", "beta", "se", "p", "p_fdr"]]
          .to_string(index=False))

    # ---------- 稳定性判定 ----------
    print("\n" + "-" * 78)
    pivot = res.pivot(index="signature", columns="model", values="beta")
    if "M1_proxy_scores" in pivot.columns and "M2_deconvolution" in pivot.columns:
        pivot["attenuation_pct"] = (1 - pivot["M2_deconvolution"] / pivot["M1_proxy_scores"]) * 100
        print("代理评分校正 → 反卷积校正 的 β 变化：")
        print(pivot.round(4).to_string())

        sig_p = res[res["model"] == "M2_deconvolution"].set_index("signature")["p_fdr"]
        survived = [s for s in sig_p.index if sig_p[s] < 0.05]
        print(f"\n反卷积校正后仍显著（FDR < 0.05）的签名：{survived if survived else '无'}")
        print("-" * 78)
        if survived:
            print("结论：头条结论成立 —— 模块升高【独立于】白细胞组成")
            print("      → 可放心使用原表述，并把反卷积结果替换进 Table 3")
        else:
            print("结论：反卷积校正后效应不再显著")
            print("      → 必须改写：模块升高【不能排除】由白细胞组成改变所驱动")
            print("      → 需报告衰减幅度，并把它列为核心局限")

    # ---------- 可直接粘贴的英文结果句 ----------
    print("\n" + "=" * 78)
    print("可直接粘贴进稿件的英文结果句（数字已填好）")
    print("=" * 78)
    for sig_name in signatures:
        sub = res[res["signature"] == sig_name]
        if sub.empty:
            continue
        d = dict(zip(sub["model"], zip(sub["beta"], sub["se"], sub["p"], sub["p_fdr"])))
        if "M2_deconvolution" not in d:
            continue
        b0, s0, p0, f0 = d.get("M0_unadjusted", (np.nan,) * 4)
        b2, s2, p2, f2 = d["M2_deconvolution"]
        n_cells = len(info["cells_used"])
        print(f"\n[{sig_name}]")
        print(f"  After adjustment for age, sex, and {n_cells} cell-type abundance "
              f"estimates obtained by deconvolution, the association with sepsis "
              f"{'remained significant' if f2 < 0.05 else 'was no longer significant'} "
              f"(beta = {b2:.3f}, SE = {s2:.3f}, P = {p2:.3e}, FDR = {f2:.3e}), "
              f"compared with beta = {b0:.3f} (P = {p0:.3e}) before adjustment.")

    print("\n" + "=" * 78)
    print(f"输出目录：{os.path.abspath(a.outdir)}")
    print("  cell_adjustment_results.csv    三种模型的完整统计表")
    print("  cell_adjustment_summary.json   样本量、使用的细胞类型、被剔除的变量")
    print("=" * 78)


if __name__ == "__main__":
    main()
