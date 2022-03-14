import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import plotly.graph_objects as go

#TODO plotting the whole grid
#todo reliable way to report final data
#todo reliable way for automated plotting

def plot_GEN (name, data):
    T_in = np.array(data ['T_in'])[722:] - 273.15
    G_in = np.array(data ['G_in'])[722:]
    T_out = np.array(data['T_out'])[722:] - 273.15
    G_out = np.array(data['G_out'])[722:]
    P = np.array(data ['P'])[722:]

    save_path = 'plots/'+name+'_T_G.png'

    t = np.linspace(0, len(G_in), 24, dtype=int)
    datelist = [(i, dt.time(i).strftime('%I'))[0] for i in range(24)]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('G_return [kg/s]', color=color)
    #ax1.plot(G_in, color=color, label= 'G_in')
    #color = 'tab:orange'
    ax1.plot(G_out, color=color, label= 'G_out')
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    color1 = 'tab:orange'
    ax2.set_ylabel('T [°C]', color=color)  # we already handled the x-label with ax1
    ax2.plot(T_in, color=color, label='T_return')
    #ax2.plot(T_out, color=color1, label='T_mandata')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xticks(t, datelist)
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()

    save_path = 'plots/'+name+'P.png'

    fig, ax = plt.subplots()
    Phi = P * 1e-6

    x = np.linspace(0, 24, len(Phi))
    ax.plot(x, Phi)
    ax.set_xlabel('t [h]')
    ax.set_ylabel('Phi [MW]')

    plt.savefig(save_path)
    plt.close()


def plotBCT(name,data):
    T_in = np.array(data['T_in_dist'])[722:] - 273.15
    G_in = np.array(data['G_in_dist'])[722:]
    T_out = np.array(data['T_out_dist'])[722:] - 273.15
    G_out = np.array(data['G_out_dist'])[722:]
    P = np.array(data['P'])[722:]

    save_path = 'plots/' + name + '_T_G.png'

    t = np.linspace(0, len(G_in), 24, dtype=int)
    datelist = [(i, dt.time(i).strftime('%I'))[0] for i in range(24)]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('G_return [kg/s]', color=color)
    # ax1.plot(G_in, color=color, label= 'G_in')
    # color = 'tab:orange'
    ax1.plot(G_in, color=color, label='G_out')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    color1 = 'tab:orange'
    ax2.set_ylabel('T [°C]', color=color)  # we already handled the x-label with ax1
    ax2.plot(T_in, color=color, label='T_return')
    ax2.plot(T_out, color=color1, label='T_mandata')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xticks(t, datelist)
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_path)
    plt.close()

    save_path = 'plots/' + name + 'P.png'

    fig, ax = plt.subplots()
    Phi = P * 1e-6

    x = np.linspace(0, 24, len(Phi))
    ax.plot(x, Phi)
    ax.set_xlabel('t [h]')
    ax.set_ylabel('Phi [MW]')

    plt.savefig(save_path)
    plt.close()

def plot_overview_T (data):

    for name, d in data.items():
        save_path = 'plots/overview_'+ name+'.png'
        T_mandata = np.array(d['T_mandata'])[722:]
        T_m = [x[4]-273.15 for x in T_mandata]
        plt.plot(T_m, label=name+'_T_m')

        T_ritorno = np.array(d['T_ritorno'])[722:]
        T_r = [x[4] - 273.15 for x in T_ritorno]
        plt.plot(T_r, label=name+'_T_r')
    plt.xlabel("time")
    plt.ylabel("Temperature [°C]")
    plt.legend()
    plt.savefig(save_path)

def plot_network(graph):
    pass

def join_grids (scenario):
    pass



if __name__ == '__main__':
    with open ('Final_reports.pickle','rb') as f:
        reports = pickle.load(f)
        f.close()

    #print (reports)
    overview = {}

    for transp_grid, data in reports.items():
        for el, d in data.items():
            #print(el, d)

            #dist grid reports
            if 'dist' in el:
                for comp , data_com in d.items():
                    if 'BCT' in comp:
                        plotBCT(comp, data_com)
                    if 'Ut' in comp:
                        pass
                    if 'Ut' not in comp and 'BCT' not in comp:
                        overview[comp] = data_com




            #generators reports
            elif 'GEN' in el:
                plot_GEN(el,d)
                pass

            #whole transport reports
        if transp_grid == 'transp':
            overview[transp_grid]=data['transp']

#overview in graph form
graph_data = '/Users/pietrorandomazzarino/Documents/DOTTORATO/CODE/Collab_Teleriscaldamento/data/CompleteNetwork_G1_D5'

plot_overview_T(overview)
