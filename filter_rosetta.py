#!/share01/hpc/guoyu/software/miniconda3/bin/python
import json
import pandas as pd
import argparse
import os
import shutil

# Standard amino acid 3-letter to 1-letter mapping
AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def extract_chain_a_seq(pdb_file):
    """Extract fast Chain A sequence (no dependencies)"""
    seq, seen = [], set()
    try:
        with open(pdb_file, 'r', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM'): continue
                chain = line[21:22].strip()
                res = line[17:20].strip()
                res_num = line[23:27].strip()
                if chain != 'A' or res not in AA_MAP: continue
                if res_num not in seen:
                    seen.add(res_num)
                    seq.append(AA_MAP[res])
        return ''.join(seq) if seq else None
    except:
        return None

def main():
    ap = argparse.ArgumentParser()
    # Core inputs
    ap.add_argument('--score', default='score.sc', help='score file (default: score.sc)')
    ap.add_argument('--pdblist', default='pdblist', help='pdblist file (default: pdblist)')
    ap.add_argument('--outdir', default='filtered_PDBs', help='PDB output dir')
    # ✅ 修改默认输出文件名：filtered_rosetta.csv
    ap.add_argument('--output', default='filtered_rosetta.csv', help='filtered output CSV')
    # 序列提取开关，默认关闭
    ap.add_argument('--extract-seq', action='store_true', help='Extract Chain A sequence (DEFAULT: disabled)')
    # 3 core filters
    ap.add_argument('--sap', type=float, default=30.0, help='max SAP score')
    ap.add_argument('--ddg', type=float, default=-44.0, help='max DDG (lower = better)')
    ap.add_argument('--cms', type=float, default=300.0, help='min CMS')
    args = ap.parse_args()

    # --------------------------
    # 1. 加载评分数据
    # --------------------------
    print(f"[1/6] Loading score file: {args.score}")
    data = []
    with open(args.score) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} total entries")

    # --------------------------
    # 2. 加载 pdblist 路径
    # --------------------------
    print(f"\n[2/6] Checking pdblist file")
    pdblist_exists = os.path.exists(args.pdblist)
    if pdblist_exists:
        print(f"✅ Found pdblist file | Loading PDB paths")
        with open(args.pdblist) as f:
            pdbs = [l.strip() for l in f if l.strip()]
        # 匹配PDB路径
        for i, row in df.iterrows():
            if i < len(pdbs):
                df.at[i, 'pdb_path'] = pdbs[i]
    else:
        print(f"ℹ️ Pdblist not found | Skipping PDB processing")

    # ====================== 输出全量数据 all_rosetta.csv ======================
    print(f"\n[3/6] Saving full data to all_rosetta.csv (all entries, no sequence)")
    all_data = df.copy()
    if 'chain_a_sequence' in all_data.columns:
        all_data = all_data.drop(columns=['chain_a_sequence'])
    all_data.to_csv('all_rosetta.csv', index=False)
    print(f"✅ all_rosetta.csv saved | Total entries: {len(all_data)}")
    # ============================================================================

    # --------------------------
    # 4. 应用过滤条件
    # --------------------------
    print(f"\n[4/6] Applying SAP/DDG/CMS filters...")
    mask = (
        (df.sap_score <= args.sap) &
        (df.ddg <= args.ddg) &
        (df.contact_molecular_surface >= args.cms)
    )
    final = df[mask].copy()
    print(f"✅ Filtering complete | {len(final)} entries passed filters")

    # --------------------------
    # 5. 仅对过滤后的条目提取序列
    # --------------------------
    print(f"\n[5/6] Sequence & PDB processing")
    do_extract = pdblist_exists and args.extract_seq
    if do_extract and not final.empty:
        print(f"🔍 Extracting sequences for filtered entries ({len(final)})...")
        final['chain_a_sequence'] = final['pdb_path'].apply(extract_chain_a_seq)
        # 保留有效序列
        final = final[final['chain_a_sequence'].notna()].copy()
        print(f"✅ Valid sequences: {len(final)} entries")
    else:
        print(f"ℹ️ Sequence extraction is DISABLED (default)")

    # --------------------------
    # 6. 复制PDB文件
    # --------------------------
    if pdblist_exists and not final.empty:
        print(f"📋 Copying {len(final)} PDB files to {args.outdir}")
        os.makedirs(args.outdir, exist_ok=True)
        for p in final.pdb_path:
            if os.path.exists(p):
                shutil.copy2(p, args.outdir)
        print(f"✅ PDB copying complete")
    else:
        print(f"ℹ️ Skipping PDB copying")

    # --------------------------
    # 7. 动态设置输出列 + 重命名
    # --------------------------
    final.rename(columns={'decoy': 'name'}, inplace=True)
    # 基础列（固定）
    base_cols = ['name', 'sap_score', 'ddg', 'contact_molecular_surface', 'pdb_path']
    # 开启序列提取则添加序列列
    if do_extract:
        output_cols = base_cols + ['chain_a_sequence']
    else:
        output_cols = base_cols
    
    final = final[output_cols].copy()

    # --------------------------
    # 8. 保存过滤后的CSV
    # --------------------------
    print(f"\n[6/6] Saving filtered results to {args.output}...")
    final.to_csv(args.output, index=False)

    # 最终总结
    print("\n" + "="*70)
    print(f'🎉 Task finished successfully!')
    print(f'📄 all_rosetta.csv: All raw entries ({len(all_data)} rows) | No sequence')
    print(f'📄 {args.output}: Filtered entries ({len(final)} rows) | Columns: {", ".join(output_cols)}')
    print(f'📊 Sequence extraction: {"ENABLED" if do_extract else "DISABLED"}')
    print("="*70)

if __name__ == '__main__':
    main()
