import numpy as np
import matplotlib.pyplot as plt

#data_folder = './experiment_results/'
task = 'halfmoon'
#flags = ['wb', 'wb_kernel', 'kernel', 'nn']
flags = ['nn']
for flag in flags:
    #fname = task+flag+'.npy'
    fname = 'adv.npy'
    [standard, at] = np.load(fname)
    ep = [0.01*i for i in range(21)]
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    axes = plt.gca()
    ymin = 0.1
    ymax = 1
    axes.set_ylim([ymin,ymax])
    l1 = ax.plot(ep, standard, marker = 's', label = 'StandardNN')
    l2 = ax.plot(ep, at, marker = 'o', label = 'ATNN')
    legend = ax.legend(loc = 'lower left', fontsize = 12)
    ax.set_ylabel('Classification Accuracy', fontsize = 18)
    ax.set_xlabel('Max $l_2$ Norm of Adv. Perturbation', fontsize = 18)
    if flag == 'wb' or flag=='kernel':
        ax.set_title('halfmoon', fontsize = 20)
    fig.tight_layout()
    plt.savefig(task+'_'+flag+'.pdf')
    plt.show()
