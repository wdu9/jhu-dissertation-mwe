import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import sys 
from HARK.utilities import make_figs
from matplotlib_config import show_plot

cwd             = os.getcwd()
folders         = cwd.split(os.path.sep)
top_most_folder = folders[-1]
if top_most_folder == 'FromPandemicCode':
    Abs_Path = cwd
    figs_dir = '../../../Figures'
    res_dir = '../Results'
else:
    Abs_Path = cwd + '\\FromPandemicCode'
    figs_dir = '../../Figures'
    res_dir = 'Results'
sys.path.append(Abs_Path)

plt.style.use('classic')

# From the data: 
#data_LorenzPts = np.array([[0, 0.01, 0.60, 3.58], [0.06, 0.63, 2.98, 11.6], [0.15, 0.92, 3.27, 10.3],\
#                           [0.03, 0.35, 1.84, 7.42]])

data_LP_popln = pd.read_csv(figs_dir + '/Data/LorenzAll.csv', sep='\t')
data_LP_byEd = pd.read_csv(figs_dir + '/Data/LorenzEd.csv', sep='\t')

# Figure labels:
mytitles = ['Dropout (9.3 pct)', 'Highschool (52.7 pct)', 'College (38 pct)', 'Population']
myYticks = [range(0,30,5), range(0,30,5),range(0,30,5),range(0,30,5)]
x_axis = np.array([20,40,60,80])


#%% Plot figure for main results or robustness case (CRRA or R)
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.5)
axs = gs.subplots(sharex=False, sharey=False)

if len(sys.argv) < 2:
    # Plot baseline figure 
    resFile = open(res_dir+'/AllResults_CRRA_2.0_R_1.01.txt', 'r')  
    model_LorenzPts = []
    
    plotWithSplurgeZero = True
    if plotWithSplurgeZero:
        resFileSplZero = open(res_dir+'/AllResults_CRRA_2.0_R_1.01_Splurge0.txt', 'r')
        model_LorenzPtsSplZero = []
    
    for line in resFile:
        if "Lorenz Points" in line:
            theLPstr = line[line.find('[')+1:line.find(']')].split(', ')
            theLPfloat = []
            for ii in range(0,len(theLPstr)):
                theLPfloat.append(float(theLPstr[ii]))
            if np.array_equal(model_LorenzPts, []):
                model_LorenzPts = np.array([theLPfloat])
            else:
                model_LorenzPts = np.append(model_LorenzPts,[theLPfloat],axis=0)
 
    if plotWithSplurgeZero:
        for line in resFileSplZero:
            if "Lorenz Points" in line:
                theLPstr = line[line.find('[')+1:line.find(']')].split(', ')
                theLPfloat = []
                for ii in range(0,len(theLPstr)):
                    theLPfloat.append(float(theLPstr[ii]))
                if np.array_equal(model_LorenzPtsSplZero, []):
                    model_LorenzPtsSplZero = np.array([theLPfloat])
                else:
                    model_LorenzPtsSplZero = np.append(model_LorenzPtsSplZero,[theLPfloat],axis=0)
 
    for row in range(2):
        for col in range(2):
            idx = col+row*(row+1)+1
            if idx < 4:
                dfToPlot = data_LP_byEd[data_LP_byEd['myEd']==idx]
                if idx == 1:
                    dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 96.5]
                else:
                    dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 90]
                axs[row,col].plot(dfToPlot['sumEdW'],dfToPlot['sumLW'], color="royalblue",
                                  linestyle='solid', linewidth=1.5, label='Data')
            else:
                dfToPlot = data_LP_popln[data_LP_popln['sumNormW'] <= 93]
                axs[row,col].plot(dfToPlot['sumNormW'],dfToPlot['sumLWall'], color="royalblue",
                                  linestyle='solid', linewidth=1.5, label='Data')
            axs[row,col].plot(x_axis, model_LorenzPts[col+row*(row+1)], color="tab:red", 
                              linestyle='dashed', linewidth=1.5,label='Model')
            if plotWithSplurgeZero:
                axs[row,col].plot(x_axis, model_LorenzPtsSplZero[col+row*(row+1)], color="tab:green", 
                                  linestyle='dotted', linewidth=1.5,label='Model - Splurge=0')
                
            axs[row,col].set_xticks(ticks=[0,20,40,60,80,100])
            axs[row,col].set_yticks(ticks=myYticks[col+row*(row+1)])
            axs[row,col].set_title(mytitles[col+row*(row+1)])
            axs[row,col].title.set_fontsize(12)
            
            if idx == 4:
                handles, labels = axs[row,col].get_legend_handles_labels()
    
    for ax in axs.flat:
        ax.set(xlabel='Percentile', ylabel='Cumulative share of wealth')
        #ax.label_outer()
    plt.rc('axes', labelsize=12)
    
    if plotWithSplurgeZero:
        lgd = fig.legend(handles, labels, loc='lower center', ncol=3, fancybox=True, shadow=False, 
                  bbox_to_anchor=(0.5, -0.01), fontsize=12)
    else:
        lgd = fig.legend(handles, labels, loc='lower center', ncol=2, fancybox=True, shadow=False, 
                  bbox_to_anchor=(0.5, -0.01), fontsize=12)
    fig.set_facecolor(color="white")
    
    plt.gcf().subplots_adjust(bottom=0.14)
#    fig.savefig('LorenzPoints_CRRA_2.0_R_1.01.pdf')
    if plotWithSplurgeZero:
        make_figs('LorenzPoints_CRRA_2.0_R_1.01_wSplZero', True , False, target_dir=figs_dir)
    else:
        make_figs('LorenzPoints_CRRA_2.0_R_1.01', True , False, target_dir=figs_dir)

elif len(sys.argv) >= 2: 
    if int(sys.argv[1]) == 1:
        # Load series for robustness figure w.r.t. the interest rate R
        resFile1 = open(res_dir+'/AllResults_CRRA_2.0_R_1.005.txt', 'r')  
        resFile2 = open(res_dir+'/AllResults_CRRA_2.0_R_1.015.txt', 'r')  
        mylabels = ['R = 0.5 %', 'R = 1.5 %']
        myFigFile = 'LorenzPoints_robustness_R'
    elif int(sys.argv[1]) == 2:
        # Load series for robustness figure w.r.t. risk aversion CRRA
        resFile1 = open(res_dir+'/AllResults_CRRA_1.0_R_1.01.txt', 'r')  
        resFile2 = open(res_dir+'/AllResults_CRRA_3.0_R_1.01.txt', 'r')  
        mylabels = ['CRRA = 1.0', 'CRRA = 3.0']
        myFigFile = 'LorenzPoints_robustness_CRRA'

    # After loading the right data, plot the robustness graph: 
    resFiles = [resFile1, resFile2]
    mylines = []
    
    for ff in range(0,2):
        model_LorenzPts = []
    
        for line in resFiles[ff]:
            if "Lorenz Points" in line:
                theLPstr = line[line.find('[')+1:line.find(']')].split(', ')
                theLPfloat = []
                for ii in range(0,len(theLPstr)):
                    theLPfloat.append(round(float(theLPstr[ii]),4))
                if np.array_equal(model_LorenzPts, []):
                    model_LorenzPts = np.array([theLPfloat])
                else:
                    model_LorenzPts = np.append(model_LorenzPts,[theLPfloat],axis=0)
        mylines.append(model_LorenzPts)
    
    for row in range(2):
        for col in range(2):
            idx = col+row*(row+1)+1
            if idx < 4:
                dfToPlot = data_LP_byEd[data_LP_byEd['myEd']==idx]
                if idx == 1:
                    dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 96]
                else:
                    dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 90]    
                axs[row,col].plot(dfToPlot['sumEdW'],dfToPlot['sumLW'], color="royalblue",
                                  linestyle='solid', linewidth=1.5, label='Data')
            else:
                dfToPlot = data_LP_popln[data_LP_popln['sumNormW'] <= 93]
                axs[row,col].plot(dfToPlot['sumNormW'],dfToPlot['sumLWall'], color="royalblue",
                                  linestyle='solid', linewidth=1.5, label='Data')
            axs[row,col].plot(x_axis, mylines[0][col+row*(row+1)], color="tab:green", 
                              linestyle='dashed', linewidth=1.5,label=mylabels[0])
            axs[row,col].plot(x_axis, mylines[1][col+row*(row+1)], color="tab:red", 
                              linestyle='dashed', linewidth=1.5,label=mylabels[1])
            axs[row,col].set_xticks(ticks=[0,20,40,60,80,100])
            axs[row,col].set_yticks(ticks=myYticks[col+row*(row+1)])
            axs[row,col].set_title(mytitles[col+row*(row+1)])
            axs[row,col].title.set_fontsize(12)
            
            if idx == 4:
                handles, labels = axs[row,col].get_legend_handles_labels()
    
    for ax in axs.flat:
        ax.set(xlabel='Percentile', ylabel='Cumulative share of wealth')
        #ax.label_outer()
    plt.rc('axes', labelsize=12)
    
    lgd = fig.legend(handles, labels, loc='lower center', ncol=2, fancybox=True, shadow=False, 
              bbox_to_anchor=(0.5, -0.005), fontsize=12)
    fig.set_facecolor(color="white")
    
    plt.gcf().subplots_adjust(bottom=0.175)
#    fig.savefig(myFigFile+'.pdf')
    make_figs(myFigFile, True , False, target_dir=figs_dir)
    
    
    #%% Plot figure of the data only for motivation slide in presentation
    plot_data_only = False
    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.5)
    axs = gs.subplots(sharex=False, sharey=False)

    if plot_data_only:
        
        for row in range(2):
            for col in range(2):
                idx = col+row*(row+1)+1
                if idx < 4:
                    dfToPlot = data_LP_byEd[data_LP_byEd['myEd']==idx]
                    if idx == 1:
                        dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 96.5]
                    else:
                        dfToPlot = dfToPlot[dfToPlot['sumEdW'] <= 90]
                    axs[row,col].plot(dfToPlot['sumEdW'],dfToPlot['sumLW'], color="royalblue",
                                      linestyle='solid', linewidth=1.5, label='Data')
                else:
                    dfToPlot = data_LP_popln[data_LP_popln['sumNormW'] <= 93]
                    axs[row,col].plot(dfToPlot['sumNormW'],dfToPlot['sumLWall'], color="royalblue",
                                      linestyle='solid', linewidth=1.5, label='Data')
                axs[row,col].set_xticks(ticks=[0,20,40,60,80,100])
                axs[row,col].set_yticks(ticks=myYticks[col+row*(row+1)])
                axs[row,col].set_title(mytitles[col+row*(row+1)])
                axs[row,col].title.set_fontsize(12)
                
                if idx == 4:
                    handles, labels = axs[row,col].get_legend_handles_labels()
        
        for ax in axs.flat:
            ax.set(xlabel='Percentile', ylabel='Cumulative share of wealth')
            #ax.label_outer()
        plt.rc('axes', labelsize=12)
        
#        lgd = fig.legend(handles, labels, loc='lower center', ncol=2, fancybox=True, shadow=False, 
#                  bbox_to_anchor=(0.5, -0.03), fontsize=12)
        fig.set_facecolor(color="white")
        
        make_figs('LorenzPoints_dataOnly', True , False, target_dir=figs_dir)

show_plot()