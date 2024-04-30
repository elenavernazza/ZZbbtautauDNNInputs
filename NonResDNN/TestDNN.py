import sys
sys.path.append('../')
from cms_runII_dnn_resonant.modules.data_import import *
from cms_runII_dnn_resonant.modules.basics import *
from cms_runII_dnn_resonant.modules.model_export import *
from cms_runII_dnn_resonant.modules.plotting import *

import copy
from typing import Callable, Tuple
import json, pdb
from prettytable import PrettyTable
import math
from sklearn.metrics import roc_auc_score
import numpy as np

def load_full_df(fy:FoldYielder) -> pd.DataFrame:
    df = fy.get_df(inc_inputs=True, verbose=False)
    # df['gen_sample']    = fy.get_column('gen_sample')
    # df['channel']       = fy.get_column('channel')
#     df['jet_cat']       = fy.get_column('jet_cat')
#     df['res_mass_orig'] = fy.get_column('res_mass_orig')
#     df['spin'] = fy.get_column('spin')
#     for c in set(df.gen_sample): df.loc[df.gen_sample == c, 'gen_sample'] = id2sample[c]
#     for c in set(df['jet_cat']): df.loc[(df.jet_cat == c), 'jet_cat'] = id2jet[c]
    return df

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

'''
python3 TestDNN.py --out DNNWeight_ZZbbtt_0 --run 0 
python3 TestDNN.py --out DNNWeight_ZbbHtt_0 --run 0
python3 TestDNN.py --out DNNWeight_ZttHbb_0 --run 0
'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",       dest="run",      default='0')
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    (options, args) = parser.parse_args()

    run_name = options.run

    basedir = os.getcwd()+'/'+options.out+'/'
    weight_dir = basedir + 'ensemble/'

    ensemble_0 = Ensemble.from_save(weight_dir + f'selected_set_0_{run_name}')
    ensemble_1 = Ensemble.from_save(weight_dir + f'selected_set_1_{run_name}')

    indir = os.getcwd()+'/'+options.out+'/DNNInputs'
    inpath = Path(indir)

    set_0_fy = FoldYielder(inpath/'test_0.hdf5', input_pipe=inpath/'input_pipe_0.pkl')
    set_1_fy = FoldYielder(inpath/'test_1.hdf5', input_pipe=inpath/'input_pipe_1.pkl')

    ensemble_0.predict(set_1_fy)
    ensemble_1.predict(set_0_fy)

    df = load_full_df(fy=set_0_fy).append(load_full_df(fy=set_1_fy))

    odir = basedir + '/TestingPerformance/'
    os.system('mkdir -p ' + odir)

    # plot_binary_class_pred(df, savename=odir+"Overall")
    # print('\nMu Tau')
    # plot_binary_class_pred(df[df.channel==1], density=True, savename=odir+"muTau")
    # print('E Tau')
    # plot_binary_class_pred(df[df.channel==2], density=True, savename=odir+"eTau")
    # print('Tau Tau')
    # plot_binary_class_pred(df[df.channel==0], density=True, savename=odir+"tauTau")

    print(" ### INFO: Producing fancy plots")

    import mplhep
    plt.style.use(mplhep.style.CMS)

    def SetStyle(ax, x_label, y_label, x_lim = None, y_lim = None, leg_title='', leg_loc='upper right'):
        leg = plt.legend(loc=leg_loc, fontsize=20, title=leg_title, title_fontsize=18)
        leg._legend_box.align = "left"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_lim: plt.xlim(x_lim)
        if y_lim: plt.ylim(y_lim)
        plt.grid()
        for xtick in ax.xaxis.get_major_ticks():
            xtick.set_pad(10)
        mplhep.cms.label(data=False, rlabel='(13 TeV) 137.57 $fb^{-1}$', fontsize=20)
    
    binning = np.linspace(0,1,101)
    bin_c = np.array((binning[:-1] + binning[1:]) / 2)

    #################################################
    # Inclusive
    #################################################

    DNNscore_sig = df[df['gen_target'] == 1]['pred']
    DNNscore_bkg = df[df['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_loc='upper center')
    plt.savefig(odir + '/DNNScore.png')
    plt.savefig(odir + '/DNNScore.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    r_h_DNNscore_bkg = 1 - i_h_DNNscore_bkg

    #################################################
    # Etau
    #################################################

    df_etau = df[df.channel==2]
    DNNscore_sig = df_etau[df_etau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_etau[df_etau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_eTau.png')
    plt.savefig(odir + '/DNNScore_eTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_etau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_etau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    r_h_DNNscore_bkg_etau = 1 - i_h_DNNscore_bkg_etau

    #################################################
    # Mutau
    #################################################

    df_mutau = df[df.channel==1]
    DNNscore_sig = df_mutau[df_mutau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_mutau[df_mutau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_muTau.png')
    plt.savefig(odir + '/DNNScore_muTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_mutau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_mutau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    r_h_DNNscore_bkg_mutau = 1 - i_h_DNNscore_bkg_mutau

    #################################################
    # Tautau
    #################################################

    df_tautau = df[df.channel==0]
    DNNscore_sig = df_tautau[df_tautau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_tautau[df_tautau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_tauTau.png')
    plt.savefig(odir + '/DNNScore_tauTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_tautau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_tautau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    r_h_DNNscore_bkg_tautau = 1 - i_h_DNNscore_bkg_tautau

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(r_h_DNNscore_bkg, i_h_DNNscore_sig, marker='o', linestyle='--', label='Inclusive', color=cmap(1/5))
    ax.plot(r_h_DNNscore_bkg_etau, i_h_DNNscore_sig_etau, marker='o', linestyle='--', label='ETau', color=cmap(2/5))
    ax.plot(r_h_DNNscore_bkg_mutau, i_h_DNNscore_sig_mutau, marker='o', linestyle='--', label='MuTau', color=cmap(3/5))
    ax.plot(r_h_DNNscore_bkg_tautau, i_h_DNNscore_sig_tautau, marker='o', linestyle='--', label='TauTau', color=cmap(4/5))
    SetStyle(ax, x_label=r"1-BKG", y_label=r"SIG", leg_loc='lower left')
    plt.savefig(odir + '/ROCcurve.png')
    plt.savefig(odir + '/ROCcurve.pdf')
    plt.close()

    # pdb.set_trace()