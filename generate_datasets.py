import pickle
from decodanda import *

# Generate behavior
n_trials_per_stimulus = 50
hit_rate = 0.75

stimulus = np.hstack([np.ones(n_trials_per_stimulus), -1 * np.ones(n_trials_per_stimulus)])
correct = 2 * (np.random.rand(2 * n_trials_per_stimulus) < hit_rate).astype(float) - 1
choice = stimulus * correct
labelsA = (stimulus + 1) / 2
labelsB = (choice + 1) / 2

# Generate visual
session_VIS = generate_synthetic_data(n_neurons=200, n_trials=100,
                                      timebins_per_trial=1,
                                      mixing_factor=0.5,
                                      rateA=0.6,
                                      rateB=0.0,
                                      labelsA=labelsA, labelsB=labelsB)
dec = Decodanda(session_VIS, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=False)
visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'], ndata=100)

pickle.dump(session_VIS, open('./datasets/VISp.pck', 'wb'))


# Generate motor
session_MO = generate_synthetic_data(n_neurons=200, n_trials=100,
                                     timebins_per_trial=1,
                                     mixing_factor=0.5,
                                     rateA=0.0,
                                     rateB=0.6,
                                     labelsA=labelsA, labelsB=labelsB)

pickle.dump(session_MO, open('./datasets/MOp.pck', 'wb'))


# Generate HPC
session_HPC = generate_synthetic_data(n_neurons=200, n_trials=100,
                                      timebins_per_trial=1,
                                      mixing_factor=0.5,
                                      rateA=0.35,
                                      rateB=0.35,
                                      labelsA=labelsA, labelsB=labelsB)
dec = Decodanda(session_HPC, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=False)

visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'], ndata=200)
plt.suptitle('Hippocampus')
plt.savefig('./img/hpc_pca.png')

f, axs = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={'width_ratios': [3, 2]})
ccgps, null_ccgps = dec.CCGP(plot=True, ax=axs[1])
perfs, null_perfs = dec.decode(0.8, plot=True, non_semantic=True, ax=axs[0], nshuffles=4)
plt.suptitle('Hippocampus')
plt.savefig('./img/hpc_decoding.png')
pickle.dump(session_HPC, open('./datasets/HPC.pck', 'wb'))

# Generate PFC
session_PFC = generate_synthetic_data(n_neurons=200, n_trials=100,
                                      timebins_per_trial=1,
                                      mixing_factor=0.5,
                                      mixed_term=0.5,
                                      rateA=0.5,
                                      rateB=0.5,
                                      labelsA=labelsA, labelsB=labelsB)

pickle.dump(session_PFC, open('./datasets/PFC.pck', 'wb'))
