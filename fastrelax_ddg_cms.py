from pyrosetta import *
from pyrosetta.rosetta import *
import os
import sys
import time
import pandas as pd
from pyrosetta.rosetta.protocols import rosetta_scripts

init( "-beta_nov16 -in:file:silent_struct_type binary -mute all" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" )

xml = os.path.join('RosettaFastRelaxUtil.xml')
objs = rosetta_scripts.XmlObjects.create_from_file(xml)
#FastRelax = objs.get_mover('FastRelax')
#ddg_filter = objs.get_filter("ddg")
#contact_molecular_surface = objs.get_filter("contact_molecular_surface")

selected1 = pd.read_csv('selected.csv')
for i in range(1, 4):
    selected1[f'ddG{i}'] = 0.0
    selected1[f'CMS{i}'] = 0.0
base_dir = '/home/guoy/Documents/toy/test/af2_predict/'
out_base_dir = '/home/guoy/Documents/toy/test/pyrosetta/'

for i in range(len(selected1)):
    pdb_name = selected1['description'][i]
    pos_dir = selected1['prefix'][i]
    input_file = os.path.join(base_dir,pos_dir,pdb_name+'.pdb')
    print(input_file)
    best_pose = None
    best_ddg = 0

    for j in range(1,4):
        FastRelax = objs.get_mover('FastRelax')
        ddg_filter = objs.get_filter("ddg")
        contact_molecular_surface = objs.get_filter("contact_molecular_surface")
        pose = pose_from_pdb(input_file)
        print('get pdb')
        print('start fastrelax')
        FastRelax.apply(pose)
        print('fastrelax end')
        ddg = ddg_filter.score(pose)
        cms = contact_molecular_surface.score(pose)
        selected1.at[i, f'ddG{j}'] = ddg
        selected1.at[i, f'CMS{j}'] = cms

        if ddg < best_ddg:
            best_ddg = ddg
            best_pose = pose.clone()
    if best_pose is not None:
        output_path = os.path.join(out_base_dir,'selected',pdb_name+'.pdb')
        best_pose.dump_pdb(output_path)

selected1.to_csv('result.csv')















