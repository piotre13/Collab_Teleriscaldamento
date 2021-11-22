import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('Final_reports.pickle', 'rb') as handle:
    reports = pickle.load(handle)

#print (reports)
for grid in reports:
    T_sub_ret = np.array(list(grid['sottostazioni'][0][1]['T_in']))-273.15
    G_BCT = np.array(list(grid['sottostazioni'][0][1]['G_out'])) *-1
    #plt.plot(T_sub_ret)
    plt.plot(G_BCT)
    plt.ylabel('T_ret_BCT')
    #TODO MAKE THE TEST WITH INTERPOLATED DATA
    plt.show()