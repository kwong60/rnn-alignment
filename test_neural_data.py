from dataset import Dataset
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform

dataset = Dataset("data/indy_20160407_02.mat")


# # neurons x bin size
# trial 1
# 282 neurons and duration = 1.25s
# print(fr_mat.shape)
# (282 x 63)
# # channel, unit 

# max trial 558
# 310 neurons, 651 
# max duration = 13.016000000000076


####### ISSUES ####### 
# trials are different durations
# active neurons in each trial differs
# large firing rates because small bin size


# stats = dataset.get_unit_statistics()
def max_over_trials(dataset):
    durations = []
    neuron_counts = []

    for i in range(0, 563):
        print(f'processing trial: {i+1}', end='\r')
        trial = dataset.get_trial(i)
        duration = trial.duration
        # print(f'trial: {i+1}, duration: {duration}')
        neurons = len(trial.spike_times)

        durations.append(duration)
        neuron_counts.append(neurons)
    # print(np.argmax(durations))
    # print(np.argmax(neuron_counts))
    max_dur = durations[np.argmax(durations)]
    max_neuron = neuron_counts[np.argmax(neuron_counts)]
    return max_dur, max_neuron

def create_fr_mat(trial_num, bin_size=0.02):
    '''
    creates matrices for a trial 
    
    :param trial: Trial class object
    :param bin_size: fixed bin size

    Returns:
        matrix (n, b)
        where n is number of neurons in that trial and b is bins computed from duration of trial
    '''

    trial = dataset.get_trial(trial_num) 
    start = trial.start_time
    end = trial.end_time

    bins = np.arange(start, end + bin_size, bin_size)
    neurons = [(c, u) for (c, u) in trial.spike_counts.keys() if u == 2 and trial.spike_counts[(c, u)] > 0]
    neuron_len = len(neurons)
    num_bins = len(bins) - 1
    fr_mat = np.zeros((neuron_len, num_bins))

    # for j, (c, u) in enumerate(trial.spike_counts.keys()):
    for j, (c, u) in enumerate(neurons):
        # if u == 1 and trial.spike_counts[(c, u)] > 0:
        
        times = trial.spike_times[(c, u)]
        
        # turn times into binned counts to add to matrix
        counts, _ = np.histogram(times, bins)
        # if i == 0 or i == 1 :
        #     print(f'neuron: {i+1}, {times}, {len(counts)}')
        fr_mat[j] = counts / bin_size
        

    return np.array(fr_mat)

# (neurons, bins)

def trial_dsim_mat(fr_matrix):
    '''
    Creates mean centered matrix for given trial firing rate matrix (filtered unit)
    
    :param fr_matrix: one trial firing rate matrix
    '''

    avg_fr = np.mean(fr_matrix, axis=1)
    std_fr = np.std(fr_matrix, axis=1)


    norm = (fr_matrix.T - avg_fr) / std_fr


    dsim_neurons = squareform(pdist(norm.T, metric="correlation"))

    plt.imshow(dsim_neurons)
    plt.imshow(dsim_neurons)
    plt.colorbar()
    plt.show()
    return dsim_neurons


for i in range(20):
    ...


fr_mat_trial1 = create_fr_mat(1)

fr_mat_trial5 = create_fr_mat(5)

print(fr_mat_trial1.shape)

one_dsim = trial_dsim_mat(fr_mat_trial1)

five_dsim = trial_dsim_mat(fr_mat_trial5)

### HIDDEN STATE 
# 32 x  3254 x 128,
# batch, max seq length, hidden_size

pca = PCA(n_components=2)
one = pca.fit_transform(one_dsim)

pca = PCA(n_components=2)

five = pca.fit_transform(five_dsim)
print(one.shape, five.shape)

r, disparity = orthogonal_procrustes(one, five)
print(round(disparity))
# trials = [create_fr_mat(i) for i in range(max_trials)]

# print(one_dsim.shape, five_dsim.shape)


def pad_trials_mat(trial_mats, max_neurons=310, max_bin=651, num_trials=563):
    '''
    pads each trial matrix from create_fr_mat() 
    to make them the same size and combines into a larger matrix
    
    :param trial_mats: list of matrices from create_fr_mat()
    :param max_neurons: int
    :param max_bin: int
    :param num_trials: int

    Returns:
        matrix of size (T_max, N_max, B_max)
    '''

    trials_fr_mat = np.full((num_trials, max_neurons, max_bin), np.nan)

    for i, mat in enumerate(trial_mats):
        # print(f'processing trial{i} out of {563}', end='\r')
        r, c = mat.shape
        trials_fr_mat[i, :r, :c] = mat
    return trials_fr_mat



# matrix = pad_trials_mat(trials)
        
# np.save("all_trials.npy", matrix)

# data = (num_trials, num_neurons, time_bins)
# data = np.load("all_trials.npy")


# init_X = data.copy()
# need to normalize firing rates 
N_max = 310
B_max = 651
T_max = 563

# X = init_X.reshape((B_max * T_max, N_max))
# reshape so (trial * bins, neurons)
# print(X.reshape(-1, N_max).shape)

# (neurons,)
# neuron_means = np.nanmean(X.reshape(-1, N_max), axis=0)
# print(neuron_means.shape)
# norm_X = X - neuron_means
# two_dim_mat = np.where(np.isnan(norm_X), neuron_means, norm_X)

# (trial*bins, neurons) = (366513, 310)
# print(two_dim_mat.shape)

##### ALL trials ####
# dsim_neurons = squareform(pdist(two_dim_mat.T, metric="correlation"))

# print(dsim_neurons.shape)

# plt.imshow(dsim_neurons)
# plt.colorbar()
# plt.show()

# ABOVE
########################################################################
# perform pca 
# d = 2
# summarize 

# max variance across time x trials in neuron space
# # patterns of neuron population activity
# pca = PCA(n_components=d)

# pca.fit(two_dim_mat)


# # # neural traj in pca space


# trial_mat = init_X[0]  # shape (num_neurons, time_bins)
# # each neuron is a row

# norm = (trial_mat.T - neuron_means) 


# # shape (time_bins, num_neurons)
# trial_mat_norm = np.where(np.isnan(norm), neuron_means, norm)


# # project trial into pca space
# trial_pca = pca.transform(trial_mat_norm)      # shape (time_bins, d)
# trial_pca = trial_pca.T 

# (d, 651)

# plt.figure(figsize=(6,5))
# plt.plot(trial_pca[0], trial_pca[1], marker='o', markersize=2)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title(f"Trial {0} trajectory (PC1 vs PC2)")
# plt.grid(True)
# plt.show()
# print(trial_pca.shape)



##### SYNTHETIC PROCRUSTES ##### 
# a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]])
# b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]])

# ax, fig = plt.subplots(())
# mtx1, mtx2, disparity = procrustes(a, b)
# print(round(disparity))











