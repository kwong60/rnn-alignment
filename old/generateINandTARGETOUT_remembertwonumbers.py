# The function generateINandTARGETOUT_remembertwonumbers generates the inputs and target outputs for training a recurrent neural network (RNN) on a simple task that requires memory.
# In this task the RNN receives two sequentially presented inputs that are each on for 10 timesteps. Let's call the values of these inputs C1 and C2.
# The RNN must remember both of these inputs. The RNN demonstrates that it has remembered these two inputs by simultaneously outputting the numbers C1 and C2 when a go-cue is presented.
# CJ Cueva 10.28.2024

import numpy as np
import torch



###############################################################################
#%% generate inputs and target outputs for training a recurrent neural network 
def generateINandTARGETOUT_remembertwonumbers(task_input_dict={}):
    #--------------------------------------------------------------------------
    #                 INPUTS to generateINandTARGETOUT
    #--------------------------------------------------------------------------
    # n_input:      number of inputs into the RNN. There are 2 inputs: The first input dimension is for the numbers to remember, the second input dimension is for the go-cue.
    # n_output:     number of outputs from the RNN. There are 2 outputs: The first output dimension is for the number inputted first, the second output dimension is for the number inputted second.   
    # n_T:          number of timesteps in a trial
    # n_trials:     number of trials
    # random_seed:  seed for random number generator
    # interval1set: interval before first number to remember
    # interval2set: interval between first and second number to remember
    # interval3set: interval between second number to remember and go-cue
    
    #--------------------------------------------------------------------------
    #                OUTPUTS from generateINandTARGETOUT
    #--------------------------------------------------------------------------
    # IN:           n_trials x n_T x n_input tensor
    # TARGETOUT:    n_trials x n_T x n_output tensor
    # output_mask:  n_trials x n_T x n_output tensor, elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)
    #--------------------------------------------------------------------------
    n_input = task_input_dict['n_input']
    n_output = task_input_dict['n_output']
    n_T = task_input_dict['n_T']
    n_trials = task_input_dict['n_trials']
    random_seed = task_input_dict['random_seed']
    interval1set = task_input_dict['interval1set']
    interval2set = task_input_dict['interval2set']
    interval3set = task_input_dict['interval3set']
    assert n_input==2, "Error: there should be 2 inputs to the RNN. 1 for the numbers to remember and 1 for the go-cue."
    assert n_output==2, "Error: there should be 2 outputs from the RNN. 1 for the first number and 1 for the second number."
    np.random.seed(random_seed)# set random seed for reproducible results 
    
    IN = np.zeros((n_trials,n_T,n_input))
    TARGETOUT = np.zeros((n_trials,n_T,n_output))
    output_mask = np.zeros((n_trials,n_T,n_output))
    constant1 = np.random.randn(n_trials)# on each trial the RNN is supposed to remember two different numbers, constant1[itrial] and constant2[itrial]
    constant2 = np.random.randn(n_trials)# on each trial the RNN is supposed to remember two different numbers, constant1[itrial] and constant2[itrial]
    for itrial in range(0,n_trials):# 0, 1, 2, ... n_trials-1
        # timesteps for important events in the trial, timestep are 1,2,3,...n_T, these ultimately need to be translated to indices 0,1,2,...n_T-1
        interval1 = interval1set[np.random.randint(interval1set.size)]# interval before first number to remember
        tstartC1 = interval1 + 1
        tendC1 = tstartC1 + 9# first stimulus is on for 10 timesteps
        interval2 = interval2set[np.random.randint(interval2set.size)]# interval between first and second number to remember
        tstartC2 = tendC1 + interval2 + 1
        tendC2 = tstartC2 + 9# second stimulus is on for 10 timesteps
        interval3 = interval3set[np.random.randint(interval3set.size)]# interval between second number to remember and go-cue
        tstartgocue = tendC2 + interval3 + 1# the go-cue is on until the end of the trial
        tendgocue = n_T
        tstartresponse = tstartgocue
        tendresponse = n_T
        
        # all timesteps from 1,2,3,...n_T are translated to indices 0,1,2,...n_T-1
        IN[itrial,(tstartC1-1):tendC1,0] = constant1[itrial]# constant1 is presented as an input to the RNN for 10 timesteps of the trial
        IN[itrial,(tstartC2-1):tendC2,0] = constant2[itrial]# constant2 is presented as an input to the RNN for 10 timesteps of the trial
        IN[itrial,(tstartgocue-1):tendgocue,1] = 1
        TARGETOUT[itrial,(tstartresponse-1):tendresponse,0] = constant1[itrial]
        TARGETOUT[itrial,(tstartresponse-1):tendresponse,1] = constant2[itrial]
        output_mask[itrial,(tstartresponse-1):tendresponse,:] = 1# elements 0(timepoint does not contribute to cost function), 1(timepoint contributes to cost function)
        
    # convert to pytorch tensors 
    dtype = torch.float32
    #IN = torch.from_numpy(IN, dtype=dtype); TARGETOUT = torch.from_numpy(TARGETOUT, dtype=dtype); 
    IN = torch.tensor(IN, dtype=dtype); TARGETOUT = torch.tensor(TARGETOUT, dtype=dtype); output_mask = torch.tensor(output_mask, dtype=dtype);
    task_output_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials}
    return IN, TARGETOUT, output_mask, task_output_dict 





#%%############################################################################
#                       test generateINandTARGETOUT
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #import os
    #figure_dir = os.path.dirname(__file__)# return the folder path for the file we're currently executing
    
    np.random.seed(123); torch.manual_seed(123)# set random seed for reproducible results
    n_input = 2# number of inputs into the RNN. There are 2 inputs: The first input dimension is for the numbers to remember, the second input dimension is for the go-cue.
    n_output = 2# number of outputs from the RNN. There are 2 outputs: The first output dimension is for the number inputted first, the second output dimension is for the number inputted second.   
    n_T = 100# number of timesteps in a trial
    n_trials = 100# number of trials
    random_seed = 1# seed for random number generator
    interval1set = np.arange(0,11)# interval before first number to remember
    interval2set = np.arange(0,11)# interval between first and second number to remember
    interval3set = np.arange(0,11)# interval between second number to remember and go-cue
    T = np.arange(0,n_T)# (n_T,)
    task_input_dict = {'n_input':n_input, 'n_output':n_output, 'n_T':n_T, 'n_trials':n_trials, 'random_seed':random_seed, 'interval1set':interval1set, 'interval2set':interval2set, 'interval3set':interval3set}
    IN, TARGETOUT, output_mask, task_output_dict = generateINandTARGETOUT_remembertwonumbers(task_input_dict)   

    

    plt.figure()# inputs and target outputs for RNN
    fontsize = 12
    for itrial in range(n_trials):
        plt.clf()
        #----colormaps----
        cool = cm.get_cmap('cool', n_input)
        colormap_input = cool(range(n_input))# (n_input, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        copper = cm.get_cmap('copper_r', n_output)
        colormap_output = copper(range(n_output))# (n_output, 4) array columns 1,2,3 are the RGB values, column 4 sets the transparency/alpha, datapoint[0] has color colormap[0,:]
        #----plot all inputs and outputs----
        ilabelsinlegend = np.round(np.linspace(0,n_input-1,5,endpoint=True))# if there are many inputs only label 5 of them in the legend
        for i in range(n_input):# 0,1,2,...n_input-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3, label=f'Input {i+1}')# label inputs 1,2,3,..
            else:
                plt.plot(T, IN[itrial,:,i], c=colormap_input[i,:], linewidth=3)# don't label these inputs
        ilabelsinlegend = np.round(np.linspace(0,n_output-1,5,endpoint=True))# if there are many outputs only label 5 of them in the legend
        for i in range(n_output):# 0,1,2,...n_output-1
            if np.isin(i,ilabelsinlegend):
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3, label=f'Output {i+1}')# label outputs 1,2,3,..
            else:
                plt.plot(T[output_mask[itrial,:,i]==1], TARGETOUT[itrial,output_mask[itrial,:,i]==1,i], c=colormap_output[i,:], linewidth=3)# don't label these outputs
        #---------------------
        plt.legend(loc='best', fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize)
        plt.title(f'Trial {itrial}', fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.xlim(left=0)
        #plt.savefig('%s/generateINandTARGETOUT_trial%g.pdf'%(figure_dir,itrial), bbox_inches='tight')# add bbox_inches='tight' to keep title from being cutoff
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.show()
        input("Press Enter to continue...")# pause the program until the user presses Enter
     
        










    