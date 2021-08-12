# Version 1.9

import json
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt



###
file_in = 'cluster_count_t.txt'
file_out = 'cluster_select_t.txt'

shuffle = 'y'   # 'y': suffle sets
manual = 'y'    # 'y': manual selection of alpha
fold_pick = 10
alpha_pick = 10**-3.2
###





with open(file_in) as f:
    data = json.load(f)

cluster = data['List']
cluster_count = data['Count']
cluster_energy = data['Energy']
sets = len(cluster_energy)



cluster_energy = np.array(cluster_energy)
axis_range = [round(min(cluster_energy))-1, round(max(cluster_energy))+1]

alpha_range = [-6, 0]
alpha_lasso = np.logspace(alpha_range[0], alpha_range[1], num=(alpha_range[1]-alpha_range[0])*10+1)



# Shuffle sets
if shuffle == 'y':
    from random import shuffle
    sets_list = [i for i in range(sets)]
    shuffle(sets_list)
    cluster_count_suffle = []
    cluster_energy_suffle = []
    for c, v in enumerate(sets_list):
        cluster_count_suffle.append([])
        cluster_count_suffle[c] = cluster_count[v]
        cluster_energy_suffle.append([])
        cluster_energy_suffle[c] = cluster_energy[v]  
    cluster_count = cluster_count_suffle
    cluster_energy = cluster_energy_suffle



# LASSO, Cross-Validation
lassocv = LassoCV(alphas=alpha_lasso, normalize=True, cv=fold_pick, max_iter=1e5)
lassocv.fit(cluster_count, cluster_energy)
lassocv_rmse = np.sqrt(lassocv.mse_path_)

print("\n#####")
print("K-folds cross validation")
print("alpha: %7.4f" %lassocv.alpha_)
print("rmse:  %7.4f" %min(lassocv_rmse.mean(axis=-1)))
print("score: %7.4f" %lassocv.score(cluster_count, cluster_energy))
print("non-zero coefficients: %i" %np.count_nonzero(lassocv.coef_))

plt.figure()
m_log_alphas = -np.log10(lassocv.alphas_)
plt.plot(m_log_alphas, lassocv_rmse, ':')
plt.plot(m_log_alphas, lassocv_rmse.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(lassocv.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
plt.xlabel('-log(alpha)'); plt.ylabel('Root-mean-square error')
plt.title('Root-mean-square error on each fold'); plt.legend()
plt.tight_layout(); plt.show()

cluster_energy_ce = lassocv.predict(cluster_count)
plt.figure()
plt.scatter(cluster_energy, cluster_energy_ce, alpha=0.5)
plt.plot(axis_range, axis_range, 'k', alpha=0.5)
plt.xlim(axis_range); plt.ylim(axis_range)
plt.gca().set_aspect('equal')
plt.xlabel('Energy, DFT'); plt.ylabel('Energy, CE')
plt.tight_layout(); plt.show()



# LASSO at specified alpha
lasso = Lasso(alpha=alpha_pick, normalize=True, max_iter=1e5)
lasso.fit(cluster_count, cluster_energy)
cluster_energy_ce = lasso.predict(cluster_count)
lasso_mse = ((np.array(cluster_energy) - np.array(cluster_energy_ce))**2).mean(axis=0)
lasso_rmse = np.sqrt(lasso_mse)

print("\n#####")
print("LASSO with specified alpha")
print("alpha: %7.4f" %alpha_pick)
print("rmse:  %7.4f" %lasso_rmse)
print("score: %7.4f" %lasso.score(cluster_count, cluster_energy))
print("non-zero coefficients: %i" %np.count_nonzero(lasso.coef_))

plt.figure()
plt.scatter(cluster_energy, cluster_energy_ce, alpha=0.5)
plt.plot(axis_range, axis_range, 'k', alpha=0.5)
plt.xlim(axis_range); plt.ylim(axis_range)
plt.gca().set_aspect('equal')
plt.xlabel('Energy (eV), DFT'); plt.ylabel('Energy (eV), CE')
plt.tight_layout(); plt.show()



cluster_coef = []
cluster_pick = []

if manual == 'y':
    cluster_coef.append(lasso.intercept_)
    cluster_coef_all = lasso.coef_

else:
    cluster_coef.append(lassocv.intercept_)
    cluster_coef_all = lassocv.coef_



cluster_nonzero = [c for c, v in enumerate(cluster_coef_all) if v != 0]
for i in cluster_nonzero:
    cluster_coef.append(cluster_coef_all[i])
    cluster_pick.append(cluster[i])



data_out = {}
data_out['Coefficient'] = cluster_coef
data_out['Cluster'] = cluster_pick

with open(file_out, 'w') as f:
    json.dump(data_out, f, indent=2, sort_keys=True)