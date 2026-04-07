#!/share01/hpc/guoyu/software/miniconda3/bin/python

import os
import sys
from glob import glob
import subprocess

import json


def write_pdblist_from_list(pdb_abs_paths, output_json="pdblist.json"):
    """
    Convert a LIST OF ABSOLUTE PDB PATHS into pdblist.json.
    
    Input:  list of absolute paths (e.g., ["/home/a.pdb", "/home/b.pdb"])
    Output: pdblist.json { "/home/a.pdb": "", "/home/b.pdb": "", ... }
    """
    # Create dict: key = pdb absolute path, value = empty string
    pdb_dict = {path: "" for path in pdb_abs_paths}
    
    # Write to JSON with pretty format
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pdb_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(pdb_dict)} PDB paths to {output_json}")
    return pdb_dict



def parse_dssp_structure(dssp_file, target_chain="A"):
    """
    Read DSSP file and return (residue_num, aa, ss_code) for target chain.
    """
    ss_list = []
    header_found = False

    with open(dssp_file, 'r') as f:
        for line in f:
            # Skip until we find the column header line
            if line.startswith("  #  RESIDUE AA STRUCTURE"):
                header_found = True
                continue

            if not header_found:
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Extract fields according to DSSP fixed-width format
            try:
                resnum = int(line[5:10].strip())    # residue number
                chain  = line[10:12].strip()        # chain ID
                aa     = line[12:14].strip()        # amino acid
                ss     = line[14:17].strip()       # secondary structure code

                # Only keep target chain
                if chain == target_chain:
                    # Take first character (H/E/S/T/G/I/B/space)
                    ss_code = ss[0] if ss else ' '
                    ss_list.append((resnum, aa, ss_code))
            except:
                continue

    return ss_list

def count_valid_helix_segments(ss_string, min_length=5):
    """
    Count how many CONSECUTIVE HELIX SEGMENTS (H) exist in a DSSP string.
    Only counts segments with ≥ min_length (default 5) H in a row.

    Args:
        ss_string: DSSP secondary structure string (e.g., " HHHSS HHHHH")
        min_length: minimum consecutive H to count as a valid helix

    Returns:
        total_valid_segments: number of valid helices
        helix_lengths: list of lengths for each valid helix
    """
    helix_segments = []
    current_h = 0

    # Loop through every character
    for char in ss_string:
        if char == 'H':
            current_h += 1
        else:
            # End of a helix segment
            if current_h >= min_length:
                helix_segments.append(current_h)
            current_h = 0  # reset

    # Check the very last segment after loop ends
    if current_h >= min_length:
        helix_segments.append(current_h)

    # Return total count + the lengths
    return len(helix_segments), helix_segments

# ==================== 仅新增：β折叠片计数函数（和螺旋函数格式完全一致）====================
def count_valid_sheet_segments(ss_string, min_length=3):
    """
    Count how many CONSECUTIVE SHEET SEGMENTS (E) exist in a DSSP string.
    Only counts segments with ≥ min_length (default 3) E in a row.
    """
    sheet_segments = []
    current_e = 0

    for char in ss_string:
        if char == 'E':
            current_e += 1
        else:
            if current_e >= min_length:
                sheet_segments.append(current_e)
            current_e = 0

    if current_e >= min_length:
        sheet_segments.append(current_e)

    return len(sheet_segments), sheet_segments

# ------------------- Usage -------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: helix_filter.py folder1 [folder2 folder3 ...]")
        print("Example: helix_filter.py ./pdbs1 ./pdbs2 ./test")
        sys.exit(1)

    passed = []
    not_passed = []
    pdbs = []
    input_folders = sys.argv[1:]

    # 🔹 进度提示：开始搜索PDB文件
    print("\n🔍 Searching for PDB files in input folders...")
    for folder in input_folders:
        search_pattern = os.path.join(folder, "*.pdb")
        pdb_files = glob(search_pattern)
        pdbs += pdb_files

    total_pdbs = len(pdbs)
    # 🔹 进度提示：显示总文件数
    print(f"📂 Found {total_pdbs} PDB files to process!\n")
    print("-" * 80)

    # 遍历处理，带进度
    for index, pdb_file in enumerate(pdbs, 1):
        pdb_abs = os.path.abspath(pdb_file)
        # 🔹 进度提示：当前处理第几个文件 + 路径
        print(f"📌 Processing [{index}/{total_pdbs}]: {pdb_abs}")

        try:
            command = "dssp -i " + pdb_file + " -o tmp.log"
            output = subprocess.check_output(command, shell=True, executable="/bin/bash")
        except subprocess.CalledProcessError as e:
            print(f"❌ DSSP command failed!")
            print(f"Error: {e.output.decode()[:100]}...")
            not_passed.append(pdb_abs)
            print("-" * 80)
            continue

        # 解析二级结构
        dssp_path = "tmp.log"
        chainA_ss = parse_dssp_structure(dssp_path, target_chain="A")
        ss_sequence = ''.join([ss for _, _, ss in chainA_ss])
        num_helix, _ = count_valid_helix_segments(ss_sequence)
        num_sheet, _ = count_valid_sheet_segments(ss_sequence)

        # 筛选判断
        if num_helix > 2 or num_sheet > 0:
            # 🔹 进度提示：通过，显示螺旋/折叠片数量
            print(f"✅ PASSED | Helices: {num_helix} | Sheets: {num_sheet}")
            passed.append(pdb_abs)
        else:
            # 🔹 进度提示：失败，显示螺旋/折叠片数量
            print(f"❌ FAILED | Helices: {num_helix} | Sheets: {num_sheet}")
            not_passed.append(pdb_abs)

        print("-" * 80)

    # 🔹 最终总结
    print("\n📊 Processing Finished!")
    print(f"✅ Total Passed: {len(passed)}")
    print(f"❌ Total Failed: {len(not_passed)}")
    print("-" * 80)

    # 保存结果
    write_pdblist_from_list(passed)
    write_pdblist_from_list(not_passed, output_json="badfold.json")
