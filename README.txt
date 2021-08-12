01. Training Data for KRA Cluster Expansion
    - Training dataset for the cluster expansion study
    - Each folder contains DFT calculation results of systems when the target VO is at starting(A), finishing(B), or transition(T) position.

02. KRA Cluster Expansion
    - Cluster expansion study using the prepared training dataset
    - step_2_count_t.py: collect cluster information (number of each cluster in the cluster pool, cluter_3atom_40A.txt) and system energies -> cluter_count.txt
    - step_3_lasso.py: select key clusters using LASSO regression and k-fold CV analysis -> cluster_select_t.txt

11. Monte Carlo Samples for DO Calculation
    - Sample configurations for the diffusivity study
    - Collect sample configurations from Monte Carlo simulations (step_4_metro.py)
    - Oxygen nonstoichiometry (delta_X) - Fe content (MC_YYYY) - Temperature (ZZZZ K)

12. DO Calculation
    - For each sample, calculate site energies using CE (step_5_energy_s.py) and transition energies using CE for KRA (step_5_energy_t.py)
    - Calculate diffusion coefficients of the oxygen vacancy by solving the master diffusion equation (step_6_diffusion.py)
    - Calculate diffusion coefficients of the oxygen ion (py_2_do.py) and ionic conductivities (py_3_sigma.py)