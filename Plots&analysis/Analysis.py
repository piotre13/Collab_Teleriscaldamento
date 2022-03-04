import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
'''transp_193']
      1: ['transp_119']
      2: ['transp_81']
      3: ['transp_93']
      4: ['transp_114']
'''



with open('Final_reports.pickle', 'rb') as handle:
    reports = pickle.load(handle)
    #handle.close()


#plot temperature and portata of generator
T_gen=[]
for x in reports['transp']['T_mandata']:
    T_gen.append(x[4])
T_gen = T_gen [722:]

T_sub1_transp = []
for x in reports['transp']['T_mandata']:
    T_sub1_transp.append(x[119])
T_sub1_transp=T_sub1_transp[722:]

T_sub1_transp_ret =[]
for x in reports['transp']['T_ritorno']:
    T_sub1_transp_ret.append(x[119])
T_sub1_transp_ret = T_sub1_transp_ret[722:]

G_sub1_transp =[]
for x in reports['transp']['G_mandata']:
    G_sub1_transp.append(x[81])
G_sub1_transp = G_sub1_transp[722:]

G_gen = []
for x in reports['transp']['G_mandata']:
    G_sub1_transp.append(x[4])
G_sub1_transp = G_sub1_transp[722:]
#NB le G sono sulle branchessssss PD


#plotting substation temperatures
t = np.linspace(0,len(T_sub1_transp),24, dtype= int)
datelist = [(i, dt.time(i).strftime('%I'))[0] for i in range(24)]


fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time (h)')
ax1.set_ylabel('T_BCT [°K]', color=color,)
ax1.plot(T_sub1_transp, color='tab:red', label = 'mandata')
ax1.plot(T_sub1_transp_ret, color=color, label = 'ritorno')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('G_BCT [kg/s]', color=color)  # we already handled the x-label with ax1
ax2.plot( G_gen, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(t,datelist)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()






#
#
# #print (reports)
# for grid in reports:
#
#     T_sub_ret = np.array(list(grid['sottostazioni'][0][1]['T_in'])) - 273.15
#     G_BCT = np.array(list(grid['sottostazioni'][0][1]['G_out'])) * -1
#     t = np.linspace(0,len(T_sub_ret[722:]),24, dtype= int)
#     datelist = [(i, dt.time(i).strftime('%I'))[0] for i in range(24)]
#
#     fig, ax1 = plt.subplots()
#     color = 'tab:blue'
#     ax1.set_xlabel('time (h)')
#     ax1.set_ylabel('T_return_BCT [°C]', color=color)
#     ax1.plot(T_sub_ret[722:], color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#     color = 'tab:orange'
#     ax2.set_ylabel('G_BCT [kg/s]', color=color)  # we already handled the x-label with ax1
#     ax2.plot( G_BCT[722:], color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
#
#     plt.xticks(t,datelist)
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.savefig('fig2.png')
#     plt.close()
#
#
#     fig, ax = plt.subplots()
#     Phi = np.array(list(grid['sottostazioni'][0][1]['P'])[722:])*-1 *1e-6
#
#     x = np.linspace(0, 24, len(Phi))
#     ax.plot(x,Phi)
#     ax.set_xlabel('t [h]')
#     ax.set_ylabel('Phi [MW]')
#
#     plt.savefig('fig1.png')

