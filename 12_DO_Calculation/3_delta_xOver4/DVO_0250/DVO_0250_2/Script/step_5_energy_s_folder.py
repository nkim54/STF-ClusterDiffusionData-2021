# Version 1.9

import os, copy, json
import numpy as np
import module_cediff as cediff



###
atom_ind_group = [[0], [1,2], [3,4]]
vac_select = 'all'

file_in = 'cluster_select_site.txt'

folder_poscar = 'POSCAR_2'
folder_site = 'Energy_Site'
###





with open(file_in) as f:
    data = json.load(f)
cluster_coef = data['Coefficient']
cluster_pick = data['Cluster']



atom_sub_oxy = -1 ###
atom_ind_oxy = atom_ind_group[atom_sub_oxy][0]
atom_ind_vac = atom_ind_group[atom_sub_oxy][1]



###
folder_list = os.listdir(folder_poscar)
if '.DS_Store' in folder_list:
    folder_list.remove('.DS_Store')
folder_list.sort()
sets = len(folder_list)
print("\nNumber of sets: %i" %sets)

for c, file_name in enumerate(folder_list):
    file_poscar = folder_poscar + '/' + file_name
    file_out = folder_site + '/' + file_name + '.txt'
    print("\nSet %i out of %i" %(c+1, sets))
###
    


    poscar = cediff.posreader(file_poscar)
    poscar = cediff.dismatcreate(poscar)
    
    atom = poscar['AtomNum']
    atom_last = list(np.cumsum(atom))
    atom_first = [0] + atom_last[0:-1]
    atom_range = [range(atom_first[i], atom_last[i]) for i in range(len(atom))]
    
    atom_oxy = atom[atom_ind_oxy]
    atom_vac = atom[atom_ind_vac]
    atom_ovo = atom_oxy + atom_vac
    
    
    
    if vac_select == 'all':
        vac_list = list(range(atom_vac))
    else:
        vac_list = vac_select
    
    
    
    cluster_list = cediff.clustercount1(cluster_pick, poscar)
    energy = cediff.clusterE(cluster_list, cluster_coef) # original energy
    
    
    
    energy_site = np.zeros((atom_vac, atom_ovo))
    
    print("\nCalculating site energies..\n")
    for cnt, val in enumerate(vac_list):
        
        vac_num = atom_first[atom_ind_vac] + val
        print("Vacancy %i out of %i" %(cnt+1, len(vac_list)))
    
    
    
        # exchange one VO <-> all O, calculate system energy 
        for i in range(atom_oxy):
            
            oxy_num = atom_first[atom_ind_oxy] + i        
            
            pos = copy.deepcopy(poscar)
            cl_list = copy.deepcopy(cluster_list)
            
            # ABO Ob Oc O | VO VOa VO -> ABO VOa Oc O | VO Ob VO -> ABO Ob VOa O | VO Oc VO
            pos['LattPnt'][oxy_num], pos['LattPnt'][vac_num] = pos['LattPnt'][vac_num], pos['LattPnt'][oxy_num]
            pos['dismat'] = cediff.dismatswap(pos['dismat'], oxy_num, vac_num)
            
            cl_list = cediff.clusterswap1(cluster_pick, pos, cl_list,
                                         atom_ind_oxy, atom_ind_vac, oxy_num, vac_num) 
        
            en = cediff.clusterE(cl_list, cluster_coef)
            energy_site[val][i] = en
    
    
    
        # for VO <-> itself and other VO, put original energy
        for i in range(atom_vac):
            energy_site[val][atom_oxy + i] = energy
    
    print("\nDone")
    
    
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    data_out = {}
    data_out['Site'] = energy_site
    with open(file_out, 'w') as f:
        json.dump(data_out, f, indent=2, sort_keys=True, cls=NumpyEncoder)