# Version 1.9

import os, copy, json
import numpy as np
import module_cediff as cediff



###
atom_ind_group = [[0], [1,2], [3,4]]
vac_select = 'all'

cut_radius = 5.1
cut_network = 3.0

file_in = 'cluster_select_trans.txt'

folder_poscar = 'POSCAR_05'
folder_trans = 'Energy_Trans'
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
    file_out = folder_trans + '/' + file_name + '.txt'
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
        
    
    
    network = np.zeros((atom_vac, atom_ovo, atom_ovo))
    energy_kra = np.zeros((atom_vac, atom_ovo, atom_ovo))
    
    print("\nCalculating transition energies..\n")
    for cnt, val in enumerate(vac_list):
        
        vac_num = atom_first[atom_ind_vac] + val
        print("Vacancy %i out of %i" %(cnt+1, len(vac_list)))
        
    
        
        # find jumping network    
        vac_num_list = list(atom_range[atom_ind_vac])
        vac_num_list.remove(vac_num) # one of VO(VO') included in jumping network
    
        count_network = 0
        for nums_1 in range(atom_ovo):
            for nums_2 in range(nums_1+1, atom_ovo):
                oxy_a = atom_first[atom_ind_oxy] + nums_1
                oxy_b = atom_first[atom_ind_oxy] + nums_2
                
                if poscar['dismat'][oxy_a][oxy_b] < cut_network:
                    network[val][nums_1][nums_2] = 1 # available network among O and VO'
                    count_network += 1
                    
                    if any(v in [oxy_a, oxy_b] for v in vac_num_list):
                        network[val][nums_1][nums_2] = 2 # blocked network by VO
                        count_network -= 1
    
    
    
        count_jump = 0
        for nums_1 in range(atom_ovo):
    
            if 1 in network[val][nums_1]:
                
                oxy_a = atom_first[atom_ind_oxy] + nums_1
                coord_a = poscar['LattPnt'][oxy_a]
                coord_vac = poscar['LattPnt'][vac_num]
                atom_first_vac = atom_first[atom_ind_vac]
                
                pos_a = copy.deepcopy(poscar)                       # AB O Oa Ob O | VO VO' VO
                pos_a['LattPnt'].remove(coord_vac)
                pos_a['LattPnt'].insert(atom_first_vac, coord_vac)  # AB O Oa Ob O | VO' VO VO
                pos_a['LattPnt'].remove(coord_a)
                pos_a['LattPnt'].insert(atom_first_vac, coord_a)    # AB O Ob O VO' | Oa VO VO
    #            cediff.poswriter('POSCAR_'+str(nums_1+1), pos_a) ###
    
    
    
                for nums_2 in range(nums_1+1, atom_ovo):
                    
                    if network[val][nums_1][nums_2] == 1: # available network
                        count_jump += 1
#                        if count_jump%10 == 0:
#                            print('Jump %i out of %i (%i/%i)' %(count_jump, count_network, cnt+1, len(vac_list)))
                            
                            
    
                        oxy_b = atom_first[atom_ind_oxy] + nums_2
                        coord_b = poscar['LattPnt'][oxy_b]
                        
                        pos_b = copy.deepcopy(pos_a)
                        pos_b['LattPnt'].remove(coord_b)
                        pos_b['LattPnt'].insert(atom_first_vac-1, coord_b) # AB O O VO' Ob | Oa VO VO
                        pos_b = cediff.dismatcreate(pos_b)
    
    
    
                        # pick atoms closer than 'cut_radius' from jumping site VO(Oa)
                        near_ind = []
                        near_coord = []
                        for num in range(sum(atom)):
                            if pos_b['dismat'][atom_first_vac][num] < cut_radius: # first VO(Oa)
                                if pos_b['dismat'][atom_first_vac-1][num] < cut_radius: # last O(Ob)
                                    near_ind.append(num)
                                    near_coord.append(pos_b['LattPnt'][num])
                                    
                        atom_near = [0]*len(atom)
                        for c, num in enumerate(near_ind):
                            for ind in range(len(atom)):
                                if num in atom_range[ind]:
                                    atom_near[ind] += 1
                        
                        atom_near[-1] += 1 ###
                        atom_near[-2] -= 1 ###
        
                        pos = copy.deepcopy(pos_b)
                        pos['AtomNum'] = atom_near
                        pos['AtomSum'] = sum(atom_near)
                        pos['LattPnt'] = near_coord
                        pos = cediff.dismatcreate(pos)
    #                    cediff.poswriter('POSCAR_'+str(nums_1+1)+'_'+str(nums_2+1), pos) ###
    
    
                        
                        cl_list = cediff.clustercount1(cluster_pick, pos)
                        
                        jump = [pos['LattPnt'].index(coord_a), pos['LattPnt'].index(coord_b)]
                        cl_list_jump = []
                        for i in range(len(cl_list)):
                            cl_list_jump.append([])
                            for cl in cl_list[i]:
                                if any(num in jump for num in cl):
                                    cl_list_jump[i].append(cl)
                        
                        cl_energy = cediff.clusterE(cl_list_jump, cluster_coef)
                        energy_kra[val][nums_1][nums_2] = cl_energy
        
        
        
        # for O and VO' <-> VO, put high energy (blocked network)
        energy_max = np.amax(energy_kra[val])
        for i in range(atom_ovo):
            for j in range(i+1, atom_ovo):
                if network[val][i][j] == 2:
                    energy_kra[val][i][j] = energy_max + 10 ###
        
        network[val] = np.maximum(network[val], network[val].transpose()) # making a symmetry matrix
        energy_kra[val] = np.maximum(energy_kra[val], energy_kra[val].transpose())
    
    print("\nDone")
    
    
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    data_out = {}
    data_out['Network'] = network
    data_out['Trans'] = energy_kra
    with open(file_out, 'w') as f:
        json.dump(data_out, f, indent=2, sort_keys=True, cls=NumpyEncoder)