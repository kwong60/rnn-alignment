import numpy as np
import torch

def generateINandTARGETOUT_kijimoto(task_input_dict={}):
    n_input = task_input_dict['n_input']
    n_output = task_input_dict['n_output']
    n_T = task_input_dict['n_T']
    n_trials = task_input_dict['n_trials']
    random_seed = task_input_dict['random_seed']
    interval1set = task_input_dict['interval1set']
    interval2set = task_input_dict['interval2set']
    interval3set = task_input_dict['interval3set']

    assert n_input == 2, "Expected 2 inputs: stimulus channel + go cue"
    assert n_output == 1, "Expected 1 output for the abstract mapping"

    np.random.seed(random_seed)

    IN = np.zeros((n_trials, n_T, n_input))
    TARGETOUT = np.zeros((n_trials, n_T, n_output))
    output_mask = np.zeros((n_trials, n_T, n_output))

    # generate random stimulus pairs for each trial
    stim1 = np.random.randn(n_trials)
    stim2 = np.random.randn(n_trials)

    # abstract mapping (placeholder, change later)
    mapped_value = stim1 - stim2

    for itrial in range(n_trials):

        interval1 = interval1set[np.random.randint(interval1set.size)]
        tstart1 = interval1 + 1
        tend1 = tstart1 + 9

        interval2 = interval2set[np.random.randint(interval2set.size)]
        tstart2 = tend1 + interval2 + 1
        tend2 = tstart2 + 9

        interval3 = interval3set[np.random.randint(interval3set.size)]
        tstart_go = tend2 + interval3 + 1
        tend_go = n_T

        # inputs
        IN[itrial, tstart1-1:tend1, 0] = stim1[itrial]
        IN[itrial, tstart2-1:tend2, 0] = stim2[itrial]
        IN[itrial, tstart_go-1:tend_go, 1] = 1

        # outputs only during go period
        TARGETOUT[itrial, tstart_go-1:tend_go, 0] = mapped_value[itrial]
        output_mask[itrial, tstart_go-1:tend_go, 0] = 1

    IN = torch.tensor(IN, dtype=torch.float32)
    TARGETOUT = torch.tensor(TARGETOUT, dtype=torch.float32)
    output_mask = torch.tensor(output_mask, dtype=torch.float32)

    task_output_dict = {
        'n_input': n_input,
        'n_output': n_output,
        'n_T': n_T,
        'n_trials': n_trials
    }

    return IN, TARGETOUT, output_mask, task_output_dict
