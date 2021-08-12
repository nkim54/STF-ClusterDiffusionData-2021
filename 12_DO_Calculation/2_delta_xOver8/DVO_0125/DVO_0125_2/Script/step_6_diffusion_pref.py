import os, json
import numpy as np
import module_cediff as cediff



###
prefactor = 1.0
###

###
atom_ind_group = [[0], [1,2], [3,4]]
vac_select = 'all'

cut_network = 3.0
temperature = [300, 600, 900, 1200, 1500] # Kelvin

folder_poscar = 'POSCAR'
folder_site = 'Energy_Site'
folder_trans = 'Energy_Trans'
file_out = 'dvo_0125.txt'
###



###
list_poscar = os.listdir(folder_poscar)
if '.DS_Store' in list_poscar:
    list_poscar.remove('.DS_Store')
list_poscar.sort()
sets = len(list_poscar)
print("\nNumber of sets: %i\n" %sets)



list_site = os.listdir(folder_site)
if '.DS_Store' in list_site:
    list_site.remove('.DS_Store')
list_site.sort()



list_trans = os.listdir(folder_trans)
if '.DS_Store' in list_trans:
    list_trans.remove('.DS_Store')
list_trans.sort()

dvo_diff = np.zeros([sets, len(temperature)])
###



atom_sub_oxy = -1 ###
atom_ind_oxy = atom_ind_group[atom_sub_oxy][0]
atom_ind_vac = atom_ind_group[atom_sub_oxy][1]

for sample in range(sets):
    file_poscar = folder_poscar + '/'+ list_poscar[sample]
    file_site = folder_site + '/' + list_site[sample]
    file_trans = folder_trans + '/' + list_trans[sample]   
       
    
    
    with open(file_site, 'r') as f:
        data_site = json.load(f)
    energy_site = data_site['Site']
    
    with open(file_trans, 'r') as f:
        data_trans = json.load(f)
    network = data_trans['Network']
    energy_kra = data_trans['Trans']
    
    
        
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
        
    
    
    diff_list = []
    diff_list_log = []
    diff_list_log_mean = []
    for vac in vac_list:
    
        # site energy
        energy_site_vac = np.array(energy_site[vac])
        energy_site_vac = energy_site_vac*(160.0/1000.0)
    
    
        
        # transition energy, kinetically resolved activation barrier
        network_vac = np.array(network[vac])
        energy_kra_vac = np.array(energy_kra[vac])
        energy_trans_vac = np.zeros((atom_ovo, atom_ovo))
        count = 0
        for i in range(atom_ovo):
            for j in range(atom_ovo):
                if network_vac[i][j] != 0:
                    if energy_kra_vac[i][j] < abs(energy_site_vac[i] - energy_site_vac[j])/2:
                        energy_trans_vac[i][j] = max(energy_site_vac[i], energy_site_vac[j])
                    else:
                        energy_trans_vac[i][j] = energy_kra_vac[i][j] + (energy_site_vac[i] + energy_site_vac[j])/2
        
        # network
        o_o_ind_list = []
        for i in range(atom_ovo):
            o_o_ind_list.append([c for c, v in enumerate(energy_trans_vac[i]) if v != 0])
        o_o_ind_list = np.array(o_o_ind_list)
        
        
        
        # displacement
        transport_vec = np.zeros((atom_ovo, atom_ovo, 3))
        count = 0
        for n in range(atom_ovo):
            for c, v in enumerate(o_o_ind_list[n]):
                vec_a = np.array(poscar['LattPnt'][atom_first[atom_ind_oxy]+n])
                vec_b = np.array(poscar['LattPnt'][atom_first[atom_ind_oxy]+v])
                
                vec = vec_b - vec_a
                for i in range(3):
                    if (vec[i]>0.5):
                        vec[i] = vec[i] - 1
                    if (vec[i]<-0.5):
                        vec[i] = vec[i] + 1
                        
                vec_cart = np.dot(vec, poscar['Base'])
                transport_vec[n][v] = vec_cart
        
        
        
        # prefactor
        pre = np.full(atom_ovo, 1.0)
        preT = np.full((atom_ovo, atom_ovo), 1.0/prefactor)
        
        
        # temperature
        kB = 8.617333e-5   # Boltzmann constant, eV/K
        beta_list = 1/(np.array(temperature)*kB)
        
        
        
        # input
        energy_min = min(energy_site_vac)
        site_ene = energy_site_vac - energy_min
        
        o_o_elist = np.zeros((atom_ovo, atom_ovo))
        for i in range(atom_ovo):
            for j in range(atom_ovo):
                if network_vac[i][j] != 0:
                    o_o_elist[i][j] = energy_trans_vac[i][j] - energy_min
        
        
        
        # diffusion coefficients
        D_T_list =  cediff.diffuser(temperature, beta_list, site_ene,
                                       o_o_ind_list, o_o_elist,
                                       pre, preT, transport_vec)
        
        diff_list.append(D_T_list)


    
    ###
    print("Set %i out of %i" %(sample+1, sets))
    dvo_avg = np.log10(np.mean(np.mean(diff_list, axis=0), axis=1)*10**4)
    dvo_diff[sample] = dvo_avg
    ###



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

data_out = {}
data_out['DVO'] = dvo_diff
with open(file_out, 'w') as f:
    json.dump(data_out, f, indent=2, sort_keys=True, cls=NumpyEncoder)