import sys
sys.path.append('../')
from cms_runII_dnn_resonant.modules.data_import import *
from cms_runII_dnn_resonant.modules.basics import *
from cms_runII_dnn_resonant.modules.model_export import *
from cms_runII_dnn_resonant.modules.plotting import *

import copy
from typing import Callable, Tuple
import json, pdb, glob
from prettytable import PrettyTable
import math
from sklearn.metrics import roc_auc_score
import numpy as np

import mplhep
plt.style.use(mplhep.style.CMS)

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
python3 TestDNN.py --out ResTest1 --run 0 
'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",       dest="run",      default='0')
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    (options, args) = parser.parse_args()

    run_name = options.run
    basedir = os.getcwd()+'/'+options.out+'/'
    odir = basedir + '/TestingPerformance/'
    os.system('mkdir -p ' + odir)

    ################################################
    print(" ### INFO: Plotting Feature Importance")
    ################################################

    if "ZZ" in options.out:
        pp = 'ZZ'; p_tt = 'Z'; p_bb = 'Z'; fancy_name = '$ZZ_{bb\\tau\\tau}$'; o_name = 'ZZbbtt'
    elif "ZbbHtt" in options.out:
        pp = 'ZH'; p_tt = 'H'; p_bb = 'Z'; fancy_name = '$Z_{bb}H_{\\tau\\tau}$'; o_name = 'ZbbHtt'
    elif "ZttHbb" in options.out:
        pp = 'ZH'; p_tt = 'Z'; p_bb = 'H'; fancy_name = '$Z_{\\tau\\tau}H_{bb}$'; o_name = 'ZttHbb'
    
    cont_feat = ['hh_kinfit_chi2', 'hh_kinfit_m', 'sv_mass', 'dR_l1_l2_x_sv_pT', 'l_1_mt', 'l_2_pT', 'dR_l1_l2',
                'dphi_sv_met', 'h_bb_mass', 'b_2_hhbtag', 'diH_mass_sv', 'dphi_hbb_sv', 'h_bb_pT', 
                'dR_l1_l2_boosted_htt_met', 'l_1_pT', 'b_1_pT', 'phi', 'costheta_l2_httmet', 'b_1_cvsb', 'b_1_cvsl',
                'boosted', 'channel', 'is_vbf', 'jet_1_quality', 'jet_2_quality', 'year', 'mass']
    
    cont_feat_name = [  fr'$\chi^{2}$(KinFit)', fr'$M_{{{pp}}}$(KinFit)', fr'$M_{{{p_tt}}}$(SVFit)', fr'$\Delta R (l_{1},l_{2}) \times p_T$(SVFit)', 
                        fr'$m_T (l_{1})$', fr'$p_T (l_{2})$',  fr'$\Delta R (l_{1},l_{2})$', fr'$\Delta\phi (MET, {{{p_tt}}}$(SVFit)$)$',
                        fr'$M ({{{p_bb}}}_{{bb}})$', fr'HHbtag$(b_{2})$', fr'$M_{{{pp}}}$(SVFit)', fr'$\Delta\phi ({{{p_bb}}}_{{bb}}, {{{p_tt}}}$(SVFit)$)$', 
                        fr'$p_T ({{{p_bb}}}_{{bb}})$', fr'$\Delta R (l_{1},l_{2}) \times (MET+{{{p_tt}}}_{{\tau\tau}})$',
                        fr'$p_T (l_{1})$', fr'$p_T (b_{1})$', fr'$\Phi$', fr'$\cos \Theta \,(l_{2}, (MET+{{{p_tt}}}_{{\tau\tau}}))$', fr'CvsB $(b_{1})$', fr'CvsL $(b_{1})$',
                        'Boosted', 'Channel', 'VBF', fr'quality $(jet_{1})$', fr'quality $(jet_{2})$', 'Year', 'Mass']
    
    feature_name_dict = dict(zip(cont_feat, cont_feat_name))

    for num in [0, 1]:
        feat_res_csv = basedir + f'train_weights_{num}/Feat_Importance.csv'
        df = pd.read_csv(feat_res_csv)
        df = df.sort_values(by='Importance', ascending=False)
        df['Feature_Name'] = df['Feature'].map(feature_name_dict)

        plt.figure(figsize=(15, 12))
        bars = plt.barh(df['Feature_Name'], df['Importance'], xerr=df['Uncertainty'], align='center', alpha=0.7, ecolor='black', capsize=5)
        plt.xlabel('Importance')
        plt.title(f'Importance via Feature Permutation ({fancy_name})')
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(odir + f'/FeatureImportance_{num}.png')
        plt.savefig(odir + f'/FeatureImportance_{num}.pdf')
        plt.xscale('log')
        plt.savefig(odir + f'/FeatureImportance_{num}_log.png')
        plt.savefig(odir + f'/FeatureImportance_{num}_log.pdf')

    ################################################
    print(" ### INFO: Plotting Training History")
    ################################################

    for num in [0, 1]:
        loss_json = basedir + f'train_weights_{num}/Loss_History.json'
        with open(loss_json, 'r') as file:
            df = json.load(file)
        
        plt.figure(figsize=(12, 12))
        cmap = plt.get_cmap('tab20')
        for i in range(0,len(df)):
            plt.plot(np.arange(0,len(df[i][0]['Training'])), df[i][0]['Training'], '.', linestyle='--', color=cmap(2*i+1))
        for i in range(0,len(df)):
            plt.plot(9*np.arange(1,len(df[i][0]['Validation'])+1), df[i][0]['Validation'], 'o', linestyle='-', color=cmap(2*i), label=f"Model {i}")
        plt.xlabel('Sub-Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid()
        plt.title(f'Loss History ({fancy_name})')
        plt.savefig(odir + f'/TrainingHistory_{num}.png')
        plt.savefig(odir + f'/TrainingHistory_{num}.pdf')

    ################################################
    print(" ### INFO: Plotting Performance")
    ################################################

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
        mplhep.cms.label(data=False, rlabel='137.1 $fb^{-1}$ (13 TeV)', fontsize=20)
    
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
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_title=fancy_name, leg_loc='upper center')
    plt.savefig(odir + '/DNNScore.png')
    plt.savefig(odir + '/DNNScore.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    # r_h_DNNscore_bkg = 1 - i_h_DNNscore_bkg

    #################################################
    # Etau
    #################################################

    df_etau = df[df.channel==2]
    DNNscore_sig = df_etau[df_etau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_etau[df_etau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_title=fancy_name, leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_eTau.png')
    plt.savefig(odir + '/DNNScore_eTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_etau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_etau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    # r_h_DNNscore_bkg_etau = 1 - i_h_DNNscore_bkg_etau

    #################################################
    # Mutau
    #################################################

    df_mutau = df[df.channel==1]
    DNNscore_sig = df_mutau[df_mutau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_mutau[df_mutau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_title=fancy_name, leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_muTau.png')
    plt.savefig(odir + '/DNNScore_muTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_mutau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_mutau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    # r_h_DNNscore_bkg_mutau = 1 - i_h_DNNscore_bkg_mutau

    #################################################
    # Tautau
    #################################################

    df_tautau = df[df.channel==0]
    DNNscore_sig = df_tautau[df_tautau['gen_target'] == 1]['pred']
    DNNscore_bkg = df_tautau[df_tautau['gen_target'] == 0]['pred']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
    ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
    SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_title=fancy_name, leg_loc='upper center')
    plt.savefig(odir + '/DNNScore_tauTau.png')
    plt.savefig(odir + '/DNNScore_tauTau.pdf')
    plt.close()

    h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
    h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
    h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
    h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
    i_h_DNNscore_sig_tautau = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
    i_h_DNNscore_bkg_tautau = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
    # r_h_DNNscore_bkg_tautau = 1 - i_h_DNNscore_bkg_tautau

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(i_h_DNNscore_bkg, i_h_DNNscore_sig, marker='o', linestyle='--', label='Inclusive', color=cmap(1/5))
    ax.plot(i_h_DNNscore_bkg_etau, i_h_DNNscore_sig_etau, marker='o', linestyle='--', label='ETau', color=cmap(2/5))
    ax.plot(i_h_DNNscore_bkg_mutau, i_h_DNNscore_sig_mutau, marker='o', linestyle='--', label='MuTau', color=cmap(3/5))
    ax.plot(i_h_DNNscore_bkg_tautau, i_h_DNNscore_sig_tautau, marker='o', linestyle='--', label='TauTau', color=cmap(4/5))
    SetStyle(ax, x_label=r"BKG Efficiency", y_label=r"SIG Efficiency", leg_title=fancy_name, leg_loc='lower right')
    plt.savefig(odir + '/ROCcurve.png')
    plt.savefig(odir + '/ROCcurve.pdf')
    plt.yscale('log')
    plt.savefig(odir + '/ROCcurve_log.png')
    plt.savefig(odir + '/ROCcurve_log.pdf')    
    plt.close()

    #################################################
    # Different masses
    #################################################

    cmap = plt.get_cmap('viridis')
    fig, AX = plt.subplots(figsize=(10,10))

    # pdb.set_trace()
    test_masses_0 = glob.glob(indir + '/test_M*_0.hdf5')
    test_masses_1 = glob.glob(indir + '/test_M*_1.hdf5')
    i = 0
    for test_0_imass, test_1_imass in zip(test_masses_0, test_masses_1):

        i = i + 1
        mass = test_0_imass.split("test_M")[1].split("_0.hdf5")[0]
        odir_imass = basedir + f'/TestingPerformance/M{mass}/'
        os.system('mkdir -p ' + odir_imass)

        print(f'\n ### Mass = {mass} GeV')

        set_0_fy_imass = FoldYielder(test_0_imass, input_pipe=inpath/'input_pipe_0.pkl')
        set_1_fy_imass = FoldYielder(test_1_imass, input_pipe=inpath/'input_pipe_1.pkl')

        ensemble_0.predict(set_0_fy_imass)
        ensemble_1.predict(set_1_fy_imass)

        df = load_full_df(fy=set_0_fy_imass).append(load_full_df(fy=set_1_fy_imass))

        DNNscore_sig = df[df['gen_target'] == 1]['pred']
        DNNscore_bkg = df[df['gen_target'] == 0]['pred']

        fig, ax = plt.subplots(figsize=(10,10))
        ax.hist(DNNscore_sig, bins=binning, density=True, linewidth=2, histtype='step', color='Blue', label='Signal')
        ax.hist(DNNscore_bkg, bins=binning, density=True, linewidth=2, histtype='step', color='Red', label='Background')
        SetStyle(ax, x_label=r"DNN Score", y_label="A.U.", leg_title=fancy_name, leg_loc='upper center')
        plt.savefig(odir_imass + '/DNNScore.png')
        plt.savefig(odir_imass + '/DNNScore.pdf')
        plt.close()

        h_DNNscore_sig, _ = np.histogram(DNNscore_sig, bins=binning)
        h_DNNscore_bkg, _ = np.histogram(DNNscore_bkg, bins=binning)
        h_DNNscore_sig_norm = h_DNNscore_sig/len(DNNscore_sig)
        h_DNNscore_bkg_norm = h_DNNscore_bkg/len(DNNscore_bkg)
        i_h_DNNscore_sig = np.array([np.sum(h_DNNscore_sig_norm[bin_c >= i]) for i in binning])
        i_h_DNNscore_bkg = np.array([np.sum(h_DNNscore_bkg_norm[bin_c >= i]) for i in binning])
        # r_h_DNNscore_bkg = 1 - i_h_DNNscore_bkg

        AX.plot(i_h_DNNscore_bkg, i_h_DNNscore_sig, marker='o', linestyle='--', color=cmap(i/len(test_masses_0)), label=f'M{mass}')
        
    SetStyle(ax, x_label=r"BKG Efficiency", y_label=r"SIG Efficiency", leg_loc='lower right')
    plt.legend(ncol=3, fontsize=17)
    plt.savefig(odir + '/ROCcurve_Mass.png')
    plt.savefig(odir + '/ROCcurve_Mass.pdf')
    plt.yscale('log')
    plt.savefig(odir + '/ROCcurve_Mass_log.png')
    plt.savefig(odir + '/ROCcurve_Mass_log.pdf')  
    plt.close()


    eos_dir = f'/eos/user/e/evernazz/www/ZZbbtautau/B2GPlots/2024_06_14/{o_name}/DNNPlots/Res'
    user = 'evernazz'
    print(f" ### INFO: Copy results to {user}@lxplus.cern.ch")
    print(f"           Inside directory {eos_dir}\n")

    # [FIXME] Work-around for mkdir on eos
    os.system(f'mkdir -p TMP_RESULTS_RES && cp index.php TMP_RESULTS_RES')
    os.system(f'cp ' + odir + f'/*.p* TMP_RESULTS_RES')
    os.system(f'rsync -rltv TMP_RESULTS_RES/* {user}@lxplus.cern.ch:{eos_dir}')
    os.system(f'rm -r TMP_RESULTS_RES')