import pickle
import numpy as np

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
                                      rateA=0.5,
                                      rateB=0.0,
                                      labelsA=labelsA, labelsB=labelsB)

pickle.dump(session_VIS, open('./datasets/VISp.pck', 'wb'))
dec = Decodanda(session_VIS, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=True)
visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'])
dec.CCGP(plot=True)
dec.decode(0.8, plot=True, non_semantic=True)

# Generate motor
session_MO = generate_synthetic_data(n_neurons=200, n_trials=100,
                                     timebins_per_trial=1,
                                     mixing_factor=0.5,
                                     rateA=0.0,
                                     rateB=0.6,
                                     labelsA=labelsA, labelsB=labelsB)

pickle.dump(session_MO, open('./datasets/MOp.pck', 'wb'))
dec = Decodanda(session_MO, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=True)
visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'])
dec.CCGP(plot=True)
dec.decode(0.8, plot=True, non_semantic=True)

# Generate HPC
session_HPC = generate_synthetic_data(n_neurons=200, n_trials=100,
                                      timebins_per_trial=1,
                                      mixing_factor=0.5,
                                      rateA=0.3,
                                      rateB=0.3,
                                      labelsA=labelsA, labelsB=labelsB)

dec = Decodanda(session_HPC, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=True)
visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'])
dec.CCGP(plot=True)
dec.decode(0.8, plot=True, non_semantic=True)
pickle.dump(session_HPC, open('./datasets/HPC.pck', 'wb'))

# Generate PFC
session_PFC = generate_synthetic_data(n_neurons=200, n_trials=100,
                                      timebins_per_trial=1,
                                      mixing_factor=0.5,
                                      mixed_term=0.5,
                                      rateA=0.5,
                                      rateB=0.5,
                                      labelsA=labelsA, labelsB=labelsB)

dec = Decodanda(session_PFC, {'stimulus': [-1, 1], 'action': [-1, 1]}, verbose=True)
visualize_PCA(dec, mean=True, names=['A>', 'A<', 'B>', 'B<'])
dec.CCGP(plot=True)
dec.decode(0.8, plot=True, non_semantic=True)
pickle.dump(session_PFC, open('./datasets/PFC.pck', 'wb'))

visualize_session(session_PFC)