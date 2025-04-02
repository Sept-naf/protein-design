import sys
# this script is used to fix residues before protein design in https://github.com/nrbennet/dl_binder_design/tree/main

def main(pdb_filename, fix_mask):
    with open(pdb_filename, 'a') as f:
        for resi in fix_mask:
            f.write('REMARK PDBinfo-LABEL:%5s FIXED\n' % (resi))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <pdb_filename> <fix_mask>")
        sys.exit(1)

    pdb_filename = sys.argv[1]
    try:
        fix_mask = list(map(int, sys.argv[2].split(',')))
    except ValueError:
        print("Error: The fix_mask should be a list of integers separated by commas.")
        sys.exit(1)

    main(pdb_filename, fix_mask)
