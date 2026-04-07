#!/share01/hpc/guoyu/software/miniconda3/bin/python

import argparse, os, shutil, pandas as pd

def parse_score(score_file):
    df = pd.read_csv(
        score_file, sep=r'\s+', skiprows=1, header=None,
        names=['binder_aligned_rmsd','pae_binder','pae_interaction',
               'pae_target','plddt_binder','plddt_target',
               'plddt_total','target_aligned_rmsd','time','pdb_name']
    )
    for col in ['plddt_binder', 'pae_interaction']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def main():
    ap = argparse.ArgumentParser(description="Copy PDBs that meet pLDDT, PAE and RMSD cut-offs.")
    # 默认评分文件：当前目录 af2.score
    ap.add_argument("--score", default="af2.score",
                    help="Path to score file (default: af2.score in current directory)")
    # 目录参数可选
    ap.add_argument("--indir", help="Directory containing PDB files (optional)")
    ap.add_argument("--outdir", help="Directory to copy filtered PDBs (optional)")
    # 筛选参数
    ap.add_argument("--plddt", type=float, default=85.0, help="Min binder pLDDT (default: 85.0)")
    ap.add_argument("--pae",   type=float, default=10.0, help="Max interaction PAE (default: 10.0)")
    # ✅ 新增：binder_aligned_rmsd 筛选，默认 < 2.0
    ap.add_argument("--rmsd",  type=float, default=2.0,  help="Max binder_aligned_rmsd (default: 2.0)")
    
    args = ap.parse_args()

    # 校验评分文件是否存在
    if not os.path.isfile(args.score):
        print(f"Error: Score file '{args.score}' not found!")
        return

    # 数据读取与预处理
    df = parse_score(args.score)
    all_out = os.path.join(os.getcwd(), "all_scores.csv")
    df.to_csv(all_out, sep="\t", index=False)
    
    mask = (df['plddt_binder'] > args.plddt) & \
           (df['pae_interaction'] < args.pae) & \
           (df['binder_aligned_rmsd'] < args.rmsd)
           
    selected_df = df.loc[mask]
    selected = selected_df['pdb_name'].tolist()

    # 打印统计信息（新增RMSD统计）
    print(f"Total designs          : {len(df)}")
    print(f"pLDDT > {args.plddt}      : {len(df[df['plddt_binder'] > args.plddt])}")
    print(f"PAE  < {args.pae}        : {len(df[df['pae_interaction'] < args.pae])}")
    print(f"RMSD < {args.rmsd}       : {len(df[df['binder_aligned_rmsd'] < args.rmsd])}")
    print(f"All 3 filters passed : {len(selected)}")

    if selected_df.empty:
        print("No designs passed filters – nothing to process.")
        return

    # 输出过滤后的评分文件到当前目录
    score_out = os.path.join(os.getcwd(), "filtered_scores.csv")
    selected_df.to_csv(score_out, sep="\t", index=False)
    print(f"\nSaved filtered scores to CURRENT WORKING DIRECTORY: {score_out}")

    # 仅当同时提供 indir 和 outdir 时，复制PDB文件
    if args.indir is not None and args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        copied_count = 0
        for stem in selected:
            src = os.path.join(args.indir, f"{stem}.pdb")
            dst = os.path.join(args.outdir, f"{stem}.pdb")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied_count += 1
            else:
                print(f"Warning: {src} not found – skipped.")
        print(f"\nCopied {copied_count} PDB files → {args.outdir}")

if __name__ == "__main__":
    main()
