import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from colabdesign import mk_afdesign_model, clear_mem
import numpy as np
import time

import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants

import argparse

def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"
  else:
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"

def add_cyclic_offset(self, offset_type=2):
  '''add cyclic offset to connect N and C term'''
  def cyclic_offset(L):
    i = np.arange(L)
    ij = np.stack([i,i+L],-1)
    offset = i[:,None] - i[None,:]
    c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
    if offset_type == 1:
      c_offset = c_offset
    elif offset_type >= 2:
      a = c_offset < np.abs(offset)
      c_offset[a] = -c_offset[a]
    if offset_type == 3:
      idx = np.abs(c_offset) > 2
      c_offset[idx] = (32 * c_offset[idx] )/  abs(c_offset[idx])
    return c_offset * np.sign(offset)
  idx = self._inputs["residue_index"]
  offset = np.array(idx[:,None] - idx[None,:])

  if self.protocol == "binder":
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset

  if self.protocol in ["fixbb","partial","hallucination"]:
    Ln = 0
    for L in self._lengths:
      offset[Ln:Ln+L,Ln:Ln+L] = cyclic_offset(L)
      Ln += L
  self._inputs["offset"] = offset

def add_rg_loss(self, weight=0.1):
  '''add radius of gyration loss'''
  def loss_fn(inputs, outputs):
    xyz = outputs["structure_module"]
    ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
    rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
    rg_th = 2.38 * ca.shape[0] ** 0.365
    rg = jax.nn.elu(rg - rg_th)
    return {"rg":rg}
  self._callbacks["model"]["loss"].append(loss_fn)
  self.opt["weights"]["rg"] = weight

#input_file = '/home/guoy/Documents/Protein_Design/cyclic/TL1A/0122/nohotspot/af2_predict/pdb.txt'
#check_point_file = '/home/guoy/Documents/Protein_Design/cyclic/TL1A/0122/nohotspot/af2_predict/check.txt'
#output_dir = '/home/guoy/Documents/Protein_Design/cyclic/TL1A/0122/nohotspot/af2_predict/'
#base_dir = '/home/guoy/Documents/Protein_Design/cyclic/TL1A/0122/nohotspot/mpnn/'
#score_file = '/home/guoy/Documents/Protein_Design/cyclic/TL1A/0122/nohotspot/af2_predict/score.sc'
parser = argparse.ArgumentParser(description="Process input files and generate PDB files.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
parser.add_argument("--check_point_file", type=str, required=True, help="Path to the checkpoint file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
parser.add_argument("--base_dir", type=str, required=True, help="Base directory for PDB files.")
parser.add_argument("--score_file", type=str, required=True, help="Path to the score file.")

args = parser.parse_args()

input_file =  args.input_file
check_point_file = args.check_point_file
output_dir = args.output_dir 
base_dir = args.base_dir
score_file = args.score_file


scores = []
f = open(input_file)
lines = f.readlines()

try:
    if os.path.isfile(check_point_file):
        with open(check_point_file, 'r') as frc:
            c_lines = frc.readlines()
            check = [item.strip() for item in c_lines]
        print("File exists and content has been read.")
    else:
        print(f"File {check_point_file} does not exist. Creating it now.")
        with open(check_point_file, 'w') as frc:
            frc.write("")  # 创建一个空文件
        check = []
        print(f"File {check_point_file} has been created.")
except Exception as e:
    print(f"An error occurred: {e}")

fc = open(check_point_file,'a')

start_time = time.time()
model = model=mk_afdesign_model('binder')

for i in range(len(lines)):
  if lines[i] in check:
    continue
  #fc.write(lines[i])
  pdb_post = lines[i].split('.')[-2]
  pdb = base_dir + pdb_post
  #model=mk_afdesign_model('binder')
  model.prep_inputs(f'{pdb}.pdb',binder_chain='A',target_chain='B',use_binder_template=True,use_multimer=True,use_initial_guess=True)
  add_cyclic_offset(model,offset_type=2)
  model.set_seq(mode='wildtype')
  model.set_opt(num_recycles=3)
  model.predict(models=[0,1],verbose=False)
  rmsd=model.aux['losses']['rmsd']
  ipae=model.aux['all']['losses']['i_pae'][0]
  model.save_pdb(f"{output_dir}{pdb_post}_{ipae:.3f}_{rmsd:.3f}_diff.pdb")
  scores.append([pdb, ipae, rmsd])
  fc.write(lines[i])

fw = open(score_file,'a')
for item in scores:
  fw.write(f'{item[0]},{item[1]},{item[2]}\n')

end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
elapsed_time = elapsed_time / 60.0
print(f"run time: {elapsed_time:.6f} min")
