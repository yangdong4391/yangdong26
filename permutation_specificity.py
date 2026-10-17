#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
M1 特异性检验：置换零模型 + 通用炎症竞争性对照
================================================================================
目的：回答审稿人最可能提出的核心质疑
  "Is the observed elevation of the death-promoting module in human sepsis
   specific to the mitoxyperilysis program, or would ANY inflammation-related
   gene set of comparable size show the same result?"

产出四项证据：
  [A] 观察效应量：模块在 脓毒症 vs 健康 中的 Cohen d（Welch t 检验）
  [B] 置换零模型：1 万次「等长 + 表达量/方差匹配」的随机基因集，给出经验 P 值
  [C] 竞争性对照：通用炎症/干扰素通路签名在同样比较中的效应量
  [D] 独立性检验：校正通用炎症后的残差是否仍与疾病相关（残差 Cohen d + 逻辑回归增量）

输入文件（需由你准备，见 README 段）：
  1. --expr    基因 × 样本 表达矩阵 CSV（index=基因符号，columns=样本ID）
  2. --meta    样本注释 CSV（index=样本ID；至少含分组列，可选 age/sex/内型/死亡率）
  3. --module  模块基因列表 TXT（每行一个基因符号）
  4. --gmt     Hallmark 基因集 GMT（可选；用于竞争性对照）

用法示例：
  python3 permutation_specificity.py \
      --expr  GSE65682_gene_matrix.csv \
      --meta  GSE65682_meta.csv \
      --module death_module.txt \
      --protective protective_module.txt \
      --group_col group --sepsis sepsis --control healthy \
      --n_perm 10000 --outdir M1_out

依赖：pandas numpy scipy statsmodels matplotlib（可选 gseapy）
================================================================================
"""
import argparse
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ============================================================================
# 基础统计工具
# ============================================================================
def cohens_d(x, y):
    """Cohen d（合并标准差版），x=病例组，y=对照组"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / s)


def zscore_rows(df):
    """按基因（行）做 z-score"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def signature_score(expr, genes, method="zmean"):
    """
    计算每个样本的签名得分。
      zmean : 基因 z-score 后取均值（默认，稳健、可复现）
      ssgsea: 调用 gseapy 的 ssGSEA（需安装 gseapy；与原稿方法一致时用这个）
    """
    present = [g for g in genes if g in expr.index]
    missing = len(genes) - len(present)
    if len(present) < 5:
        raise ValueError(f"匹配到的基因过少（{len(present)}），请检查基因符号是否为人类 HGNC 大写格式")

    if method == "ssgsea":
        try:
            import gseapy as gp
        except ImportError:
            print("[警告] 未安装 gseapy，自动改用 zmean 方法")
            method = "zmean"

    if method == "zmean":
        score = zscore_rows(expr.loc[present]).mean(axis=0)
    else:
        import gseapy as gp
        gs = {"SIG": present}
        res = gp.ssgsea(data=expr, gene_sets=gs, outdir=None, min_size=5,
                        max_size=5000, permutation_num=0, no_plot=True, processes=4)
        score = res.res2d.loc["SIG"]
        score.index = [str(i) for i in score.index]
        score = score.reindex(expr.columns)

    return score.astype(float), len(present), missing


# ============================================================================
# [B] 表达量/方差匹配的置换零模型
# ============================================================================
def build_strata(expr, n_mean_bins=10, n_sd_bins=4):
    """按 (平均表达十分位 × 表达方差四分位) 对基因分层，用于匹配抽样"""
    mean_expr = expr.mean(axis=1)
    sd_expr = expr.std(axis=1, ddof=1)
    b_mean = pd.qcut(mean_expr.rank(method="first"), n_mean_bins, labels=False)
    b_sd = pd.qcut(sd_expr.rank(method="first"), n_sd_bins, labels=False)
    return (b_mean.astype(int).astype(str) + "_" + b_sd.astype(int).astype(str))


def expression_matched_permutation(expr, module_genes, n_perm=10000, seed=2026):
    """
    抽取 n_perm 个随机基因集，满足：
      - 与模块等长
      - 与模块在「表达量 × 方差」分层上的构成完全一致
    这样得到的零分布排除了"高表达/高变异基因天然更容易出大效应量"的混杂。
    """
    present = [g for g in module_genes if g in expr.index]
    strata = build_strata(expr)
    target = strata.loc[present].value_counts()

    pool = {}
    for k in target.index:
        idx = strata.index[strata == k].to_numpy()
        if len(idx) < target[k]:
            raise ValueError(f"分层 {k} 的候选基因不足（需要 {target[k]}，池内 {len(idx)}）")
        pool[k] = idx

    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=object)
    for i in range(n_perm):
        genes = []
        for k, cnt in target.items():
            genes.extend(rng.choice(pool[k], size=int(cnt), replace=False).tolist())
        draws[i] = genes
    return draws


def permutation_null(expr, meta, module_genes, group_col, sepsis_lbl, control_lbl,
                     n_perm=10000, seed=2026, score_method="zmean", verbose=True):
    """计算随机基因集的 Cohen d 零分布"""
    sepsis_ids = meta.index[meta[group_col].astype(str) == sepsis_lbl]
    control_ids = meta.index[meta[group_col].astype(str) == control_lbl]
    keep = [c for c in expr.columns if c in set(sepsis_ids) | set(control_ids)]
    if len(keep) < 20:
        raise ValueError("表达矩阵列名与 meta 的样本ID匹配过少，请检查两者是否对齐")

    expr_keep = expr[keep]
    is_sep = expr_keep.columns.isin(sepsis_ids)

    # 观察值
    obs_score, n_present, n_missing = signature_score(expr_keep, module_genes, score_method)
    x = obs_score[is_sep].dropna()
    y = obs_score[~is_sep].dropna()
    d_obs = cohens_d(x, y)
    t_obs, p_obs = stats.ttest_ind(x, y, equal_var=False)

    # 零分布
    draws = expression_matched_permutation(expr_keep, module_genes, n_perm=n_perm, seed=seed)
    d_null = np.empty(n_perm)
    for i, genes in enumerate(draws):
        s = zscore_rows(expr_keep.loc[genes]).mean(axis=0) if score_method == "zmean" else \
            signature_score(expr_keep, genes, score_method)[0]
        d_null[i] = cohens_d(s[is_sep], s[~is_sep])
        if verbose and (i + 1) % 1000 == 0:
            print(f"    置换进度 {i+1}/{n_perm}")

    # 单侧经验 P（观察值是否比随机更大）
    p_emp_one = (np.sum(d_null >= d_obs) + 1) / (n_perm + 1)
    # 双侧：|d| 不小于观察 |d| 的比例
    p_emp_two = (np.sum(np.abs(d_null) >= abs(d_obs)) + 1) / (n_perm + 1)
    z_emp = (d_obs - np.mean(d_null)) / np.std(d_null, ddof=1)

    return dict(d_obs=d_obs, p_obs=float(p_obs), n_present=n_present, n_missing=n_missing,
                d_null=d_null, p_emp_one=float(p_emp_one), p_emp_two=float(p_emp_two),
                z_emp=float(z_emp), null_mean=float(np.mean(d_null)),
                null_sd=float(np.std(d_null, ddof=1)),
                null_q95=float(np.quantile(d_null, 0.95)),
                null_q99=float(np.quantile(d_null, 0.99)),
                n_sepsis=int(is_sep.sum()), n_control=int((~is_sep).sum()))


# ============================================================================
# [C] 竞争性对照：通用炎症通路
# ============================================================================
DEFAULT_COMPETITORS = {
    "HALLMARK_INFLAMMATORY_RESPONSE": None,   # 由 GMT 提供
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": None,
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": None,
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": None,
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": None,
}


def read_gmt(path):
    gs = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gs[parts[0]] = [g for g in parts[2:] if g]
    return gs


def competitive_control(expr, meta, gmt_path, group_col, sepsis_lbl, control_lbl,
                        score_method="zmean"):
    """通用炎症/干扰素通路在同样比较中的效应量，作为"通用炎症天花板"参照"""
    if not gmt_path or not os.path.exists(gmt_path):
        return None
    gs = read_gmt(gmt_path)
    want = [k for k in DEFAULT_COMPETITORS if k in gs]
    if not want:
        return None

    sepsis_ids = meta.index[meta[group_col].astype(str) == sepsis_lbl]
    control_ids = meta.index[meta[group_col].astype(str) == control_lbl]
    keep = [c for c in expr.columns if c in set(sepsis_ids) | set(control_ids)]
    expr_keep = expr[keep]
    is_sep = expr_keep.columns.isin(sepsis_ids)

    rows = []
    for name in want:
        s, n_present, _ = signature_score(expr_keep, gs[name], score_method)
        x = s[is_sep].dropna()
        y = s[~is_sep].dropna()
        d = cohens_d(x, y)
        _, p = stats.ttest_ind(x, y, equal_var=False)
        rows.append(dict(signature=name, n_genes=n_present, cohens_d=d, p=p))
    return pd.DataFrame(rows)


# ============================================================================
# [D] 独立性检验：校正通用炎症后残差是否仍有疾病信号
# ============================================================================
def residual_independence(expr, meta, module_genes, competitor_genes,
                          group_col, sepsis_lbl, control_lbl,
                          covars=None, score_method="zmean"):
    """
    1) 模块得分 ~ 通用炎症得分（+ 可选协变量）取残差
    2) 检验残差在 脓毒症 vs 健康 中的 Cohen d 与 Welch P
    3) 逻辑回归：sepsis ~ 炎症 + 协变量   vs   + 模块得分，报告增量效应与似然比检验
    """
    sepsis_ids = meta.index[meta[group_col].astype(str) == sepsis_lbl]
    control_ids = meta.index[meta[group_col].astype(str) == control_lbl]
    keep = [c for c in expr.columns if c in set(sepsis_ids) | set(control_ids)]
    expr_keep = expr[keep]
    is_sep = expr_keep.columns.isin(sepsis_ids)

    mod_score, n_present, _ = signature_score(expr_keep, module_genes, score_method)
    inf_score, _, _ = signature_score(expr_keep, competitor_genes, score_method)

    df = pd.DataFrame({"module": mod_score, "inflam": inf_score})
    df["group"] = np.where(expr_keep.columns.isin(sepsis_ids), 1, 0)
    if covars:
        for c in covars:
            if c in meta.columns:
                df[c] = meta.loc[expr_keep.columns, c].values
    df = df.dropna()
    use_covars = [c for c in (covars or []) if c in df.columns]

    # 残差
    X = sm.add_constant(df[["inflam"] + use_covars])
    resid = sm.OLS(df["module"], X).fit().resid
    d_res = cohens_d(resid[df["group"] == 1], resid[df["group"] == 0])
    _, p_res = stats.ttest_ind(resid[df["group"] == 1], resid[df["group"] == 0], equal_var=False)

    # 逻辑回归增量
    base_cols = ["inflam"] + use_covars
    Xb = sm.add_constant(df[base_cols])
    m0 = sm.Logit(df["group"], Xb).fit(disp=0)
    Xf = sm.add_constant(df[base_cols + ["module"]])
    m1 = sm.Logit(df["group"], Xf).fit(disp=0)
    lr_stat = 2 * (m1.llf - m0.llf)
    lr_p = stats.chi2.sf(lr_stat, 1)

    # 相关性
    r = float(np.corrcoef(df["module"], df["inflam"])[0, 1])
    collinear = abs(r) > 0.95
    if collinear:
        print("    [警告] 模块与通用炎症签名相关系数 |r| > 0.95，存在严重共线；"
              "逻辑回归的 OR 可能数值不稳定，请以『残差 Cohen d』为准。")

    return dict(
        collinear_warning=collinear,
        n=int(len(df)), n_sepsis=int((df["group"] == 1).sum()),
        n_control=int((df["group"] == 0).sum()),
        r_module_inflam=r,
        resid_cohens_d=float(d_res), resid_p=float(p_res),
        module_or_per_sd=float(np.exp(m1.params["module"])),
        module_or_ci=[float(np.exp(m1.conf_int().loc["module", 0])),
                      float(np.exp(m1.conf_int().loc["module", 1]))],
        module_logit_p=float(m1.pvalues["module"]),
        lr_stat=float(lr_stat), lr_p=float(lr_p),
    )


# ============================================================================
# 作图
# ============================================================================
def plot_null(d_null, d_obs, p_emp, out_png, out_pdf, title="Death-promoting module"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ax.hist(d_null, bins=80, color="#9fb3c8", edgecolor="white", linewidth=0.3,
            label=f"Expression-matched random gene sets (n={len(d_null)})")
    ax.axvline(d_obs, color="#c0392b", lw=2.2,
               label=f"Observed module  d = {d_obs:.2f}")
    ax.axvline(np.quantile(d_null, 0.95), color="#34495e", ls="--", lw=1.2,
               label="95th percentile of null")
    ax.set_xlabel("Cohen d  (sepsis vs healthy controls)")
    ax.set_ylabel("Number of random gene sets")
    ax.set_title(f"{title}:\nspecificity against expression-matched null", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.98, 0.92, f"Empirical $P$ = {p_emp:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fdf2e9", ec="#c0392b", lw=0.8))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


# ============================================================================
# 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="M1 特异性检验")
    ap.add_argument("--expr", required=True, help="基因×样本 表达矩阵 CSV")
    ap.add_argument("--meta", required=True, help="样本注释 CSV")
    ap.add_argument("--module", required=True, help="模块基因列表 TXT（每行一个基因）")
    ap.add_argument("--protective", default=None, help="保护模块基因列表 TXT（可选）")
    ap.add_argument("--gmt", default=None, help="Hallmark GMT（可选，用于竞争性对照）")
    ap.add_argument("--competitor", default="HALLMARK_INFLAMMATORY_RESPONSE",
                    help="用于残差校正的通用炎症基因集名称")
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--sepsis", default="sepsis")
    ap.add_argument("--control", default="healthy")
    ap.add_argument("--covars", default="", help="逗号分隔的协变量列名，如 age,sex")
    ap.add_argument("--score_method", default="zmean", choices=["zmean", "ssgsea"])
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default="M1_out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    covars = [c.strip() for c in a.covars.split(",") if c.strip()]

    print("[1/5] 读取数据")
    expr = pd.read_csv(a.expr, index_col=0)
    meta = pd.read_csv(a.meta, index_col=0)
    module = [l.strip() for l in open(a.module) if l.strip()]
    print(f"    表达矩阵 {expr.shape}；样本注释 {meta.shape}；模块基因 {len(module)}")

    print(f"[2/5] 置换检验（{a.n_perm} 次，表达量/方差匹配）")
    res_main = permutation_null(expr, meta, module, a.group_col, a.sepsis, a.control,
                                n_perm=a.n_perm, seed=a.seed, score_method=a.score_method)

    results = {"module": {
        k: v for k, v in res_main.items() if k != "d_null"
    }}
    results["module"]["null_q95"] = res_main["null_q95"]
    results["module"]["null_q99"] = res_main["null_q99"]

    plot_null(res_main["d_null"], res_main["d_obs"], res_main["p_emp_one"],
              os.path.join(a.outdir, "Fig_M1_null_distribution.png"),
              os.path.join(a.outdir, "Fig_M1_null_distribution.pdf"))

    if a.protective and os.path.exists(a.protective):
        print("[3/5] 保护模块置换检验")
        prot = [l.strip() for l in open(a.protective) if l.strip()]
        r = permutation_null(expr, meta, prot, a.group_col, a.sepsis, a.control,
                             n_perm=a.n_perm, seed=a.seed + 1, score_method=a.score_method,
                             verbose=False)
        results["protective_module"] = {k: v for k, v in r.items() if k != "d_null"}
        np.save(os.path.join(a.outdir, "null_d_protective.npy"), r["d_null"])

    print("[4/5] 通用炎症竞争性对照")
    comp = competitive_control(expr, meta, a.gmt, a.group_col, a.sepsis, a.control,
                               a.score_method)
    if comp is not None:
        comp.to_csv(os.path.join(a.outdir, "competitive_signatures.csv"), index=False)
        print(comp.to_string(index=False))
        results["competitors"] = comp.to_dict(orient="records")
        pool = read_gmt(a.gmt)
    else:
        pool = None
        print("    未提供有效的 GMT 文件，跳过竞争性对照（强烈建议补上）")

    print("[5/5] 独立性检验（校正通用炎症后的残差）")
    if pool and a.competitor in pool:
        ind = residual_independence(expr, meta, module, pool[a.competitor],
                                    a.group_col, a.sepsis, a.control,
                                    covars=covars, score_method=a.score_method)
        results["independence"] = ind
    else:
        print("    缺少通用炎症基因集，跳过独立性检验")

    # 保存
    with open(os.path.join(a.outdir, "M1_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    np.save(os.path.join(a.outdir, "null_d_module.npy"), res_main["d_null"])

    # ---------- 结果解读 ----------
    print("\n" + "=" * 72)
    print("M1 结果摘要")
    print("=" * 72)
    m = results["module"]
    print(f"模块可测基因数            : {m['n_present']} / 缺失 {m['n_missing']}")
    print(f"样本量                    : 脓毒症 {m['n_sepsis']} vs 对照 {m['n_control']}")
    print(f"观察 Cohen d              : {m['d_obs']:.3f}  (Welch P = {m['p_obs']:.3e})")
    print(f"匹配零分布                : mean {m['null_mean']:.3f} ± SD {m['null_sd']:.3f}"
          f"；95% 分位 {m['null_q95']:.3f}")
    print(f"经验 P（单侧）            : {m['p_emp_one']:.4f}")
    print(f"标准化效应 z              : {m['z_emp']:.2f}")
    print("-" * 72)
    if m["p_emp_one"] < 0.05:
        print("结论：模块升高【显著优于】表达量匹配的随机基因集")
        print("      → 支持机制特异性，可按原结论撰写，并把本图作为补充图")
    else:
        print("结论：模块升高【未超过】随机基因集所能达到的水平")
        print("      → 必须改写结论：该模块在人血中的升高主要反映通用炎症/干扰素程序，")
        print("        不能作为 mitoxyperilysis 的机制特异标志物。")
        print("      → 文章定位应转为「平移失败的负面证据 + 方法学警示」，")
        print("        这对投 Cell Death Discovery / PLOS ONE / PeerJ 反而是加分项。")
    if "independence" in results:
        i = results["independence"]
        print("-" * 72)
        print(f"模块 vs 通用炎症相关性    : r = {i['r_module_inflam']:.3f}")
        print(f"校正后残差 Cohen d        : {i['resid_cohens_d']:.3f} (P = {i['resid_p']:.3e})")
        print(f"校正后模块 OR/SD          : {i['module_or_per_sd']:.3f} "
              f"[{i['module_or_ci'][0]:.3f}–{i['module_or_ci'][1]:.3f}], "
              f"P = {i['module_logit_p']:.3e}")
        print(f"似然比检验                : chi2 = {i['lr_stat']:.2f}, P = {i['lr_p']:.3e}")
    print("=" * 72)
    print(f"输出目录：{os.path.abspath(a.outdir)}")
    print("  Fig_M1_null_distribution.png/.pdf   主图（零分布 vs 观察值）")
    print("  M1_results.json                     全部数值结果")
    print("  null_d_module.npy                   零分布原始数值（供复现）")


if __name__ == "__main__":
    main()
