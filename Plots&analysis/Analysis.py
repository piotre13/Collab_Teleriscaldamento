import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
'''   0: [transp_193']
      1: ['transp_119']
      2: ['transp_81']
      3: ['transp_93']
      4: ['transp_114']
'''



with open('Final_reports_test.pickle', 'rb') as handle:
    reports = pickle.load(handle)
    #handle.close()

# def check(data):
#     subs = ['transp_119', 'transp_81', 'transp_93', 'transp_114', 'transp_193']
#     for sub in subs:
#         id = int(sub.split('_')[-1])

#plot temperature and portata of generator
T_gen=[]
for x in reports['transp']['T_mandata']:
    T_gen.append(x[4])
T_gen = np.array(T_gen[722:])

T_gen_ret=[]
for x in reports['transp']['T_ritorno']:
    T_gen_ret.append(x[4])
T_gen_ret = np.array(T_gen_ret[722:] )

G_gen = []
for x in reports['transp']['G_mandata']:
    G_gen.append(x[2])
G_gen = np.array(G_gen[721:])

G_gen_ret = []
for x in reports['transp']['G_ritorno']:
    G_gen_ret.append(x[2])
G_gen_ret = np.array(G_gen_ret[722:])

P_gen = (G_gen * 4186 * (T_gen - T_gen_ret))*1e-6


#NB le G sono sulle branchessssss PD



t = np.linspace(0,len(T_gen),24, dtype= int)
datelist = [(i, dt.time(i).strftime('%I'))[0] for i in range(24)]


#PLOTTING generators data
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time (h)')
ax1.set_ylabel('T [°K]', color=color,)
ax1.plot(T_gen_ret, color=color, label='T_generator_ritorno')
ax1.plot(T_gen, color='tab:red', label = 'T_generator_mandata')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('G [kg/s]', color=color)  # we already handled the x-label with ax1
ax2.plot( G_gen, color=color,label = 'G_gen_mandata')
ax2.plot( G_gen_ret, color=color,label = 'G_gen_ritorno')
#ax2.plot( G_sub1_dist, color='tab:green', label='G_BCT1_mandata')
# ax2.plot( G_sub2_dist, label='G_BCT2_mandata')
# ax2.plot( G_sub3_dist, label='G_BCT3_mandata')
# ax2.plot( G_sub4_dist, label='G_BCT4_mandata')

ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(t,datelist)
plt.legend()
plt.title('Power Plant plot')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.close()



T_sub1_transp = []
for x in reports['transp']['T_mandata']:
    T_sub1_transp.append(x[119])
T_sub1_transp= T_sub1_transp[722:]

T_sub1_transp_ret =[]
for x in reports['transp']['T_ritorno']:
    T_sub1_transp_ret.append(x[119])
T_sub1_transp_ret = T_sub1_transp_ret[722:]

T_sub2_transp = []
for x in reports['transp']['T_mandata']:
    T_sub2_transp.append(x[81])
T_sub2_transp=T_sub2_transp[722:]

T_sub2_transp_ret =[]
for x in reports['transp']['T_ritorno']:
    T_sub2_transp_ret.append(x[81])
T_sub2_transp_ret = T_sub2_transp_ret[722:]




G_sub1_transp =[]
for x in reports['transp']['G_mandata']:
    G_sub1_transp.append(x[193])
G_sub1_transp = G_sub1_transp[722:]

G_sub1_dist = []
for x in reports['dist_0']['G_mandata']:
    G_sub1_dist.append(x[2])
G_sub1_dist = G_sub1_dist[722:]

G_sub2_dist = []
for x in reports['dist_1']['G_mandata']:
    G_sub2_dist.append(x[2])
G_sub2_dist = G_sub2_dist[722:]

G_sub3_dist = []
for x in reports['dist_2']['G_mandata']:
    G_sub3_dist.append(x[2])
G_sub3_dist = G_sub3_dist[722:]

G_sub4_dist = []
for x in reports['dist_3']['G_mandata']:
    G_sub4_dist.append(x[2])
G_sub4_dist = G_sub4_dist[722:]

G_sub5_dist = []
for x in reports['dist_4']['G_mandata']:
    G_sub5_dist.append(x[2])
G_sub5_dist = G_sub5_dist[722:]

#PLOTTING SUBSTATIONS DATA
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time (h)')
ax1.set_ylabel('T [°K]', color=color,)
#ax1.plot(T_sub1_transp, color='tab:red', label='T_BCT1_mandata')
ax1.plot(T_sub1_transp, color=color, label = 'T_BCT_sub119')
#ax1.plot(T_sub1_transp_ret, color='tab:red', label = 'T_BCT_sub119_ret')
plt.legend()
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('G [kg/s]', color=color)  # we already handled the x-label with ax1
#ax2.plot( G_sub2_dist, color=color,label = 'G_BCT_sub119')
#ax2.plot( G_sub3_dist, color='tab:green',label = 'G_BCT_sub81')

ax2.plot( G_sub1_dist, color='tab:green', label='G_BCT1_mandata')
# ax2.plot( G_sub2_dist, label='G_BCT2_mandata')
# ax2.plot( G_sub3_dist, label='G_BCT3_mandata')
# ax2.plot( G_sub4_dist, label='G_BCT4_mandata')

ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(t,datelist)
plt.legend()
plt.title('substations plot')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.close()





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

