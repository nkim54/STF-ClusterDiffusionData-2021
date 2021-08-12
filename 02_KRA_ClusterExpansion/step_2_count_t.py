# Version 1.9

import os, copy, json
import numpy as np
import module_cediff as cediff



###
folder = 'Set_Trans'
cut_radius = 5.1

file_poscar = 'POSCAR_A_VO'
file_oszicar_a = 'OSZICAR_A'
file_oszicar_b = 'OSZICAR_B'
file_oszicar_t = 'OSZICAR_T'

file_in = 'cluster_3atom_40A_edit.txt'
file_out = 'cluster_count_t.txt'
###





with open(file_in) as f:
    data = json.load(f)
cluster = data['List']



folder_list = os.listdir(folder)
if '.DS_Store' in folder_list:
    folder_list.remove('.DS_Store')
folder_list.sort()
sets = len(folder_list)
print("\nNumber of sets: %i" %sets)



# collect energies of each set
cluster_energy = []
print("\nCollecting energies..")
for cnt, val in enumerate(folder_list):
    os.chdir(folder+'/'+val)
    
    cmd = "grep E0 " + file_oszicar_a + " | tail -1 | awk '{printf \"%f\", $5}'"
    cl_energy_a = eval(os.popen(cmd).read())
    cmd = "grep E0 " + file_oszicar_b + " | tail -1 | awk '{printf \"%f\", $5}'"
    cl_energy_b = eval(os.popen(cmd).read())
    cmd = "grep E0 " + file_oszicar_t + " | tail -1 | awk '{printf \"%f\", $5}'"
    cl_energy_t = eval(os.popen(cmd).read())

    cl_energy = cl_energy_t - (cl_energy_a + cl_energy_b)/2 # kinetically resolved activation barrier 

    if cl_energy < 0: # if KRA is less than 0, set KRA to 0
        print("set %i out of %i, KRA less than 0" %(cnt+1, sets))
        cl_energy = 0

    cluster_energy.append(cl_energy)    
    os.chdir('../..')
print("Done")



# count numbers of each cluster
cluster_count = []
print("\nCounting clusters..\n")
for cnt, val in enumerate(folder_list):
    os.chdir(folder+'/'+val)
    print("Set %i out of %i" %(cnt+1, sets))
    
    poscar = cediff.posreader(file_poscar)
    poscar = cediff.dismatcreate(poscar)
    
    atom = poscar['AtomNum']
    atom_last = list(np.cumsum(atom))
    atom_first = [0] + atom_last[0:-1]
    atom_range = [range(atom_first[i], atom_last[i]) for i in range(len(atom))]



    # pick atoms closer than 'cut_radius' from jumping sites atoms, one VO and one O
    atom_first_vac = atom_first[-1] ### first VO among VOs
    near_ind = []
    near_coord = []
    for num in range(sum(atom)):
        if poscar['dismat'][atom_first_vac][num] < cut_radius: # first VO
            if poscar['dismat'][atom_first_vac-1][num] < cut_radius: # last O
                near_ind.append(num)
                near_coord.append(poscar['LattPnt'][num])

    atom_near = [0]*len(atom)
    for c, num in enumerate(near_ind):
        for ind in range(len(atom)):
            if num in atom_range[ind]:
                atom_near[ind] += 1
    
#    atom_near[-1] += 1 ###
#    atom_near[-2] -= 1 ###
    
    pos = copy.deepcopy(poscar)
    pos['AtomNum'] = atom_near
    pos['AtomSum'] = sum(atom_near)
    pos['LattPnt'] = near_coord
    pos = cediff.dismatcreate(pos)
#    vp.poswriter('POSCAR_A_VO_Near', pos) ###



    cl_list = cediff.clustercount1(cluster, pos)

    # collect cluster lists containing jumping sites atoms, one VO and one O
    coord_vac = poscar['LattPnt'][atom_first_vac] # VO
    coord_oxy = poscar['LattPnt'][atom_first_vac -1] # O
    jump = [pos['LattPnt'].index(coord_vac), pos['LattPnt'].index(coord_oxy)]
    cl_list_jump = []
    for i in range(len(cl_list)):
        cl_list_jump.append([])
        for cl in cl_list[i]:
            if any(num in jump for num in cl):
                cl_list_jump[i].append(cl)             



    cl_count = cediff.countCluster(cl_list_jump)
    cluster_count.append(cl_count)    
    
    os.chdir('../..')
print("\nDone")



data['Energy'] = cluster_energy
data['Count'] = cluster_count

with open(file_out, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)