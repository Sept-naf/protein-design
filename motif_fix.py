from glob import glob

def process_pdb(input_file_path):
    # 打开并读取 PDB 文件
    with open(input_file_path, 'r') as file:
        pdb_lines = file.readlines()

    # 找出 A 链中非 GLY 的残基编号
    non_gly_residues = set()
    for line in pdb_lines:
        if line.startswith('ATOM'):
            # 提取链标识
            chain = line[21].strip()
            # 提取残基类型
            res_name = line[17:20].strip()
            # 提取残基编号
            res_seq = line[22:26].strip()
            
            if chain == 'A' and res_name != 'GLY':
                non_gly_residues.add(res_seq)

    # 在文件末尾添加 REMARK 信息
    with open(input_file_path, 'a') as file:
        for res in sorted(non_gly_residues, key=lambda x: int(x)):
            # 生成格式化后的 REMARK 行
            remark_line = f"REMARK PDBinfo-LABEL:%5s FIXED\n" % res
            file.write(remark_line)

# 使用方法：替换 'your_pdb_file.pdb' 为你的 PDB 文件路径
pdbs = glob('*/*.pdb')
for pdb in pdbs:
    process_pdb(pdb)
