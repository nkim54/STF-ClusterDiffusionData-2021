import json
import numpy as np
import matplotlib.pyplot as plt



########## Setting
Fe = [1/343, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
n_Osite = 192

file_folder = '1_delta_0/DVO_Ref/dvo_'
filename = 'do_1_delta_0'
n_VO = np.array(Fe) * 32 # ref:32, oxy_1:24, oxy_2:16

# file_folder = '2_delta_xOver8/DVO_Oxy_1/dvo_'
# filename = 'do_2_delta_xOver8'
# n_VO = np.array(Fe) * 24 

# file_folder = '3_delta_xOver4/DVO_Oxy_2/dvo_'
# filename = 'do_3_delta_xOver4'
# n_VO = np.array(Fe) * 16 

data_Fe = [1, 2, 3, 4, 5, 6, 7, 8]
data_T = [300, 600, 900, 1200, 1500]
data_n = 20

dvo_1fe = [-15.12, -8.39, -6.20, -5.24, -4.70]
T = 1000/np.array(data_T)
##########



########## Plot setting
pl_m = 'x'
pl_l1 = '-'; pl_l2 = '--'

pl_x = [0, 1]; pl_xt = 0.125
pl_y = [-22, -4]; pl_yt = 2

arr_t = [600, 1500]; arr_tt = 100
arr_l = ['600','','','900','','','1200','','','1500']
arr_x = [0.6, 1.7]; arr_xt = 0.1
arr_y = [-16, -4]; arr_yt = 2
########## 



########## Data import
def logavg(log):
    return np.log10(np.nansum(10**log)/len(log))

data_dvo_raw = np.zeros((len(data_Fe), len(data_T)*data_n, len(data_T)))
for i in range(len(data_Fe)):
    file_name = '{:04d}'.format(int(data_Fe[i]/8*1000))
    with open(file_folder+file_name+'.txt', 'r') as f:
        data = json.load(f)
    data_dvo_raw[i,:,:] = np.array(data['DVO'])

data_dvo = np.zeros((len(data_Fe), len(data_T)))
for i in range(len(data_Fe)):
    
    data_dvo_1 = np.concatenate([data_dvo_raw[i, 0:10,0], data_dvo_raw[i,50:60,0]])
    data_dvo_2 = np.concatenate([data_dvo_raw[i,10:20,1], data_dvo_raw[i,60:70,1]])
    data_dvo_3 = np.concatenate([data_dvo_raw[i,20:30,2], data_dvo_raw[i,70:80,2]])
    data_dvo_4 = np.concatenate([data_dvo_raw[i,30:40,3], data_dvo_raw[i,80:90,3]])
    data_dvo_5 = np.concatenate([data_dvo_raw[i,40:50,4], data_dvo_raw[i,90:100,4]])

    data_dvo[i] = [logavg(data_dvo_1), logavg(data_dvo_2), 
                   logavg(data_dvo_3), logavg(data_dvo_4), 
                   logavg(data_dvo_5)]
    
dvo = np.insert(data_dvo, 0, dvo_1fe, axis=0)
##########



########## DVO to DO
dvo2do = np.log10(n_VO / (n_Osite - n_VO))
do = dvo + np.array([dvo2do,]*len(data_T)).transpose()
##########



########## DO Arrhenius plot
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(T, do[8], marker=pl_m, linestyle=pl_l1, label='[Fe]=1.000', color='red')
ax.plot(T, do[7], marker=pl_m, linestyle=pl_l2, label='[Fe]=0.875', color='orange')
ax.plot(T, do[6], marker=pl_m, linestyle=pl_l1, label='[Fe]=0.750', color='orange')
ax.plot(T, do[5], marker=pl_m, linestyle=pl_l2, label='[Fe]=0.625', color='green')
ax.plot(T, do[4], marker=pl_m, linestyle=pl_l1, label='[Fe]=0.500', color='green')
ax.plot(T, do[3], marker=pl_m, linestyle=pl_l2, label='[Fe]=0.375', color='blue')
ax.plot(T, do[2], marker=pl_m, linestyle=pl_l1, label='[Fe]=0.250', color='blue')
ax.plot(T, do[1], marker=pl_m, linestyle=pl_l2, label='[Fe]=0.125', color='black')
ax.plot(T, do[0], marker=pl_m, linestyle=pl_l1, label='[Fe]=0.003', color='black')

ax.set_xlim(arr_x)
ax.set_ylim(arr_y)
ax.set_xticks(np.linspace(arr_x[0], arr_x[1], round((arr_x[1]-arr_x[0])/arr_xt+1)))
ax.set_yticks(np.linspace(arr_y[0], arr_y[1], round((arr_y[1]-arr_y[0])/arr_yt+1)))
ax.set_xlabel('1000/T (1/K)')
ax.set_ylabel('DO (cm2/s)')

ax2 = ax.twiny()
ax2.set_xlim(arr_x)
ax2.set_xticks(1000/np.linspace(arr_t[0], arr_t[1], round((arr_t[1]-arr_t[0])/arr_tt+1)))
ax2.set_xticklabels(arr_l)
ax2.set_xlabel('T (K)')

ax.grid(axis='y')
ax.legend(loc='lower left')
fig.savefig(filename, bbox_inches='tight')
##########