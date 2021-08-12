# Version 1.9

import os, random, copy, json, time, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import module_cediff as cediff



###
firstrun_or_not = 1 # 0:first run, 1:re-run
#file_poscar_abo = 'POSCAR_cubic'
file_poscar_re = 'POSCAR/POSCAR_Finish_9'
folder = '09'
temperature = 300.0 # Kelvin

atom_name = ['Sr', 'Ti', 'Fe', 'O', 'VO']
atom_group = [[64], [16,48], [180,12]]

step = 60000
step_save = 5000
step_cut = 10000

file_in = 'cluster_select_site.txt'
file_metro = 'metro_all.txt'
file_metro_sample = 'metro_sample.txt'
###





with open(file_in) as f:
    data = json.load(f)
cluster_coef = data['Coefficient']
cluster_pick = data['Cluster']



kB = 8.617333e-5 # Boltzmann constant, eV/K
kBT = kB * temperature
if temperature == 0:
    kBT = 0.00001



# for the first run, poscar is randomly generated
if firstrun_or_not == 0:
    atom_group_sum = [sum(atom_group[i]) for i in range(len(atom_group))]
    atom_group_last = list(np.cumsum(atom_group_sum))
    atom_group_first = [0] + atom_group_last[0:-1]
    atom_group_range = [range(atom_group_first[i], atom_group_last[i]) for i in range(len(atom_group_sum))]
    
    pos = cediff.posreader(file_poscar_abo)
    coord = []
    for sub, group in enumerate(atom_group): 
        
        # atom with no substitution
        if len(group) == 1:
            for num in atom_group_range[sub]:
                coord.append(pos['LattPnt'][num])
        
        # atom with substitution
        else:
            at_list = list(atom_group_range[sub])
            at_list_pick = random.sample(at_list, group[1])
            at_list_pick.sort()
            for num in at_list_pick:
                at_list.remove(num)
            for num in at_list:
                coord.append(pos['LattPnt'][num])
            for num in at_list_pick:
                coord.append(pos['LattPnt'][num])
    
    atom = list(np.concatenate(atom_group))
    poscar = copy.deepcopy(pos)
    poscar['AtomNum'] = atom
    poscar['EleName'] = atom_name
    poscar['EleNum'] = len(atom)
    poscar['LattPnt'] = coord
    poscar = cediff.dismatcreate(poscar)

# if not the first run, use the last poscar from the previous run
elif firstrun_or_not == 1:
    poscar = cediff.posreader(file_poscar_re)
    poscar = cediff.dismatcreate(poscar)



atom = poscar['AtomNum']
atom_last = list(np.cumsum(atom))
atom_first = [0] + atom_last[0:-1]

# atom_group -> atom_ind -> atom_ind_sub : [[32],[20,12],[90,6]] -> [[0],[1,2],[3,4]] -> [[1,2],[3,4]]
index = 0
atom_ind_group = []
for sub, group in enumerate(atom_group):
    atom_ind_group.append([])
    for i in range(len(group)):
        atom_ind_group[sub].append(index)
        index += 1
atom_ind_sub = [group for sub, group in enumerate(atom_ind_group) if len(group) > 1]



cluster_list = cediff.clustercount1(cluster_pick, poscar)
cluster_energy = cediff.clusterE(cluster_list, cluster_coef)

pos = copy.deepcopy(poscar)
cl_list = copy.deepcopy(cluster_list)
cl_energy = cluster_energy



print("\nMonte Carlo at %i K\n" %temperature)
time_start = time.time()

os.mkdir(folder)
step_energy = []
step_energy_sample = []

cediff.poswriter(folder+'/'+'POSCAR_Start', pos) # first poscar 
for count in range(step+1):    
    pos_new = copy.deepcopy(pos)
    cl_list_new = copy.deepcopy(cl_list)
    
    
    
    # switch B <-> B' and O <-> VO
    for sub in range(len(atom_ind_sub)):
        
        num_1 = random.randint(atom_first[atom_ind_sub[sub][0]], atom_last[atom_ind_sub[sub][0]]-1)
        num_2 = random.randint(atom_first[atom_ind_sub[sub][1]], atom_last[atom_ind_sub[sub][1]]-1)       
        pos_new['LattPnt'][num_1], pos_new['LattPnt'][num_2] = pos_new['LattPnt'][num_2], pos_new['LattPnt'][num_1]
        
        pos_new['dismat'] = cediff.dismatswap(pos_new['dismat'], num_1, num_2)
        cl_list_new = cediff.clusterswap1(cluster_pick, pos_new, cl_list_new,
                                         atom_ind_sub[sub][0], atom_ind_sub[sub][1], num_1, num_2)    
    
    cl_energy_new = cediff.clusterE(cl_list_new, cluster_coef)    
#    cl_energy_delta = cl_energy_new - cl_energy
    
    
    
    ##########
    cl_energy_delta = (cl_energy_new - cl_energy) * 0.16 # meV/atom -> eV/supercell
    ##########
    
    
    
    # Metropolis–Hastings algorithm
    accept = 1
    if cl_energy_delta > 0:
        prob = math.exp(-cl_energy_delta/kBT)
        prob_rand = random.random()
        if prob < prob_rand:
            accept = 0
        
    if accept == 1:
        pos = copy.deepcopy(pos_new)
        cl_list = copy.deepcopy(cl_list_new)
        cl_energy = cl_energy_new    
        step_energy.append(cl_energy)
   
    elif accept == 0:
        step_energy.append(cl_energy)
    
    
    
    # saving poscar on every 'step_save' steps
    if count%step_save == 0:
        print("Step %i out of %i" %(count, step))
        step_energy_sample.append(cl_energy)
        cediff.poswriter(folder+'/'+'POSCAR_'+str('{:05d}'.format(count))+'_'+str('{:02d}'.format(int(cl_energy)))+str(int((cl_energy-int(cl_energy))*1000)), pos)
#        cediff.poswriter(folder+'/'+'POSCAR_'+str('{:05d}'.format(count))+'_'+str(round(cl_energy,2)), pos)



cediff.poswriter(folder+'/'+'POSCAR_Finish', pos) # last poscar

print("\nDone (%i seconds)" %(time.time()-time_start))
#print("Checking numbers of each cluster for the last step")
#print(cediff.countCluster(cediff.clustercount1(cluster_pick, pos)))
#print(cediff.countCluster(cl_list)) # Work done correctly if these two lines are the same



step_energy_sample = step_energy_sample[int(step_cut/step_save+1):] # list of sample energies after 'step_cut' steps

with open(folder+'/'+file_metro, 'w') as f:
    json.dump(step_energy, f, indent=2, sort_keys=True)

with open(folder+'/'+file_metro_sample, 'w') as f:
    json.dump(step_energy_sample, f, indent=2, sort_keys=True)



kde = stats.gaussian_kde(step_energy_sample)
kde_x = np.linspace(min(step_energy_sample)-0.5, max(step_energy_sample)+0.5, 100)
axis_range = [round(min(step_energy_sample)-0.5), round(max(step_energy_sample)+0.5)]

plt.figure()
plt.hist(step_energy_sample, bins=20, density=1, rwidth=0.9, alpha=0.5)
plt.plot(kde_x, kde(kde_x), 'k', linewidth=2)
plt.xlim(axis_range); plt.ylim(0.0, 2.0)
plt.xticks(range(int(axis_range[0]), int(axis_range[1])+1, 1))
plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
plt.xlabel('Energy (eV)'); plt.ylabel('Probability density')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout(); plt.show()

plt.figure()
plt.plot(range(0, step+1, 1), step_energy)
plt.xlim(0, step); plt.ylim(axis_range)
plt.yticks(range(int(axis_range[0]), int(axis_range[1])+1, 1))
plt.xlabel('Steps'); plt.ylabel('Energy (eV)')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout(); plt.show()