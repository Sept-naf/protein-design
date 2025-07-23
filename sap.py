from pyrosetta import *
from pyrosetta.rosetta import *
import os
import sys
import time
import pandas as pd
from pyrosetta.rosetta.protocols import rosetta_scripts

init( "-beta_nov16_cart -beta_nov16")

xml = os.path.join('macro.xml')
objs = rosetta_scripts.XmlObjects.create_from_file(xml)
#pcm = objs.get_mover('pcm')
#mini = objs.get_mover('minimize_interface')
#ddg_filter = objs.get_filter("ddg")
#contact_molecular_surface = objs.get_filter("contact_molecular_surface")


#input_files = sys.argv[1]
#output_file = sys.argv[2]
output_base = '/home/guoy/Documents/Protein_Design/cyclic/TSLP/receptor/pyrosetta_motif/sap_result'

#f = open(input_files)
fw = open('sap.txt', 'w')
#lines = f.readlines()

data = pd.read_csv('info.csv')


for line in data['PDB']:
    try:
        input_file = line.strip()
        pose = pose_from_pdb(input_file)
        pcm = objs.get_mover('pcm')
        mini = objs.get_mover('minimize_interface')
        ddg_filter = objs.get_filter("ddg")
        contact_molecular_surface = objs.get_filter("contact_molecular_surface")
        sap_metric = objs.get_simple_metric('sap_score')
        pcm.apply(pose)
        mini.apply(pose)
        pcm.apply(pose)

        ddg = ddg_filter.score(pose)
        cms = contact_molecular_surface.score(pose)
        sap = sap_metric.calculate(pose)

        print(sap)
        #break
        output_path = output_base + line.split('/')[-1][:-5] + '_sap.pdb'
        pose.dump_pdb(output_path)


        fw.write(line)
        fw.write(',')
        fw.write(str(ddg))
        fw.write(',')
        fw.write(str(cms))
        fw.write(',')
        fw.write(str(sap))
        fw.write('\n')
    except:
        print(f'{input_file} failed')
fw.close()
