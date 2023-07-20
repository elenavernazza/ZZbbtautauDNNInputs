import pandas as pd
import numpy as np
import os, sys, glob, json
import h5py
import pickle
import uproot
import matplotlib
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)
    
from sklearn.model_selection import train_test_split

from cms_runII_dnn_resonant.modules.data_import import *
from cms_runII_dnn_resonant.modules.basics import *
from cms_runII_dnn_resonant.modules.model_export import *
from cms_runII_dnn_resonant.modules.features import *

from lumin.plotting import plot_settings
import seaborn as sns
from lumin.plotting.plot_settings import PlotSettings
from lumin.plotting.data_viewing import plot_feat

def read_root_file(filename, tree_name, features):
    root_file = uproot.open(filename)
    tree = root_file[tree_name]
    return tree.arrays(features, library="pd")

def check_weights(df:pd.DataFrame) -> None:
    v = []
    for c in df.Class.unique():
        print('Class ', c)
        for m in df.pairType.unique():
            v.append(df.loc[(df.Class==c) & (df.pairType==m), 'weight'].sum())
            print(m, 'sum', v[-1])
    print(f' Channel std {np.std(v):.2f}')

def balance_weights(df:pd.DataFrame) -> None:
    df_copy = df.copy()
    print('\n Initial weight sums')
    check_weights(df_copy)
    df_copy['original_weight'] = df_copy['weight']
    for c in df_copy['pairType'].unique():
        for t in df_copy['Class'].unique():
            df_copy.loc[(df_copy['Class'] == t) & (df_copy['pairType'] == c), 'weight'] \
                /= np.sum(df_copy.loc[(df_copy['Class'] == t) & (df_copy['pairType'] == c), 'weight'])
    print('\n Final weight sums')
    check_weights(df_copy)
    return df_copy

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    data_dir = '/data_CMS/cms/vernazza/cmt/Categorization/ul_2018_ZZ_v10/'
    stat_dir = '/data_CMS/cms/vernazza/cmt/MergeCategorizationStats/ul_2018_ZZ_v10/'

    cont_feat = ['dnn_CvsB_b1', 'dnn_CvsL_b1', 'ZZKinFit_chi2', 'ZZKinFit_mass', 'Ztt_mass',
                'dnn_dR_l1_l2_x_sv_pT', 'dnn_dau1_mt', 'dau2_pt', 'dnn_dR_l1_l2', 'dnn_dphi_sv_met', 'Zbb_mass',
                'dnn_HHbtag_b2', 'ZZ_svfit_mass', 'dnn_dphi_Zbb_sv', 'Zbb_pt', 'dnn_dR_l1_l2_boosted_Ztt_met',
                'dau1_pt', 'dnn_bjet1_pt', 'dnn_Phi', 'dnn_costheta_l2_Zttmet']
    cat_feat  = ['isBoosted', 'pairType', 'dnn_deepFlav1', 'dnn_deepFlav2', 'VBFjet1_JetIdx']
    weights   = ['genWeight', 'puWeight', 'prescaleWeight', 'trigSF', 'L1PreFiringWeight_Nom', 'PUjetID_SF']

    features = cont_feat + cat_feat + weights + ['event']

    ######################### Import imputs #########################

    sig_sample = 'zz_sl_signal'
    sig_XS = 5.52 * 0.046
    files_sig = glob.glob(data_dir + sig_sample + '/cat_ZZ_elliptical_cut_80_sr/prod_DNN_Ellipse80_SR/data_*.root')
    print(" ### INFO: Reading signal samples for", sig_sample)
    data_frames = [read_root_file(filename, 'Events', features) for filename in files_sig]
    df_sig = pd.concat(data_frames, ignore_index=True)
    stat_file = stat_dir + sig_sample + '_aux/prod_230718/stats.json'
    with open(stat_file, "r") as f: 
        data = json.load(f)
        nevents = data['nevents']
        nweightedevents = data['nweightedevents']
    df_sig['xs']         = sig_XS
    df_sig['nev']        = nevents
    df_sig['nev_w']      = nweightedevents
    df_sig['gen_weight'] = df_sig['genWeight'] * df_sig['puWeight']
    df_sig['cor_weight'] = df_sig['prescaleWeight'] * df_sig['trigSF'] * df_sig['L1PreFiringWeight_Nom'] * df_sig['PUjetID_SF']
    df_sig['weight']     = sig_XS/nweightedevents * df_sig['gen_weight'] * df_sig['cor_weight']
    df_sig['sample']     = sig_sample
    del data_frames

    bkg_samples_dict = { 
        'dy': 6077.22,
        'ggf_sm': 0.03105,
        'st_antitop': 80.95,
        'st_top': 136.02,
        'st_tw_antitop': 35.85,
        'st_tw_top': 35.85,
        'tt_dl': 88.29,
        'tt_fh': 377.96,
        'tth_bb': 0.2953,
        'tth_nonbb': 0.17996,
        'tth_tautau': 0.031805,
        'tt_sl': 365.34, 
        'wjets': 61526.7, 
        'zz_dl': 1.26, 
        'zz_fh': 3.262, 
        'zz_lnu': 0.564, 
        'zz_qnu': 4.07, 
        'zz_sl_background': 5.52*0.954,
        'zzz': 0.0147
    }
    
    df_all_bkg = pd.DataFrame()
    for bkg_sample in bkg_samples_dict.keys():
        files_bkg = glob.glob(data_dir + bkg_sample + '/cat_ZZ_elliptical_cut_80_sr/prod_DNN_Ellipse80_SR/data_*.root')
        # if '/data_CMS/cms/vernazza/cmt/PreprocessRDF/ul_2018_ZZ_v10_backup/wjets/cat_base_selection/prod_DNN_Ellipse80/data_2.root' in files_bkg:
        #     files_bkg.remove('/data_CMS/cms/vernazza/cmt/PreprocessRDF/ul_2018_ZZ_v10_backup/wjets/cat_base_selection/prod_DNN_Ellipse80/data_2.root')
        # if '/data_CMS/cms/vernazza/cmt/PreprocessRDF/ul_2018_ZZ_v10_backup/wjets/cat_base_selection/prod_DNN_Ellipse80/data_1.root' in files_bkg:
        #     files_bkg.remove('/data_CMS/cms/vernazza/cmt/PreprocessRDF/ul_2018_ZZ_v10_backup/wjets/cat_base_selection/prod_DNN_Ellipse80/data_1.root')
        print(" ### INFO: Reading background samples for", bkg_sample)
        data_frames = [read_root_file(filename, 'Events', features) for filename in files_bkg]
        df_bkg = pd.concat(data_frames, ignore_index=True)
        stat_file = stat_dir + bkg_sample + '_aux/prod_230718/stats.json' if bkg_sample != 'dy' else stat_dir + 'dy_nlo_aux/prod_230718/stats.json'
        with open(stat_file, "r") as f: 
            data = json.load(f)
            nevents = data['nevents']
            nweightedevents = data['nweightedevents']
        df_bkg['xs']         = bkg_samples_dict[bkg_sample]
        df_bkg['nev']        = nevents
        df_bkg['nev_w']      = nweightedevents
        df_bkg['gen_weight'] = df_bkg['genWeight'] * df_bkg['puWeight']
        df_bkg['cor_weight'] = df_bkg['prescaleWeight'] * df_bkg['trigSF'] * df_bkg['L1PreFiringWeight_Nom'] * df_bkg['PUjetID_SF']
        df_bkg['weight']     = bkg_samples_dict[bkg_sample]/nweightedevents * df_bkg['gen_weight'] * df_bkg['cor_weight']
        df_bkg['sample']     = bkg_sample
        df_all_bkg = pd.concat([df_all_bkg, df_bkg], ignore_index=True)
        del data_frames

    df_sig.insert(0, 'Class', 0, True)
    df_all_bkg.insert(0, 'Class', 1, True)

    Events = pd.concat([df_sig, df_all_bkg])

    ######################### Print statistics #########################

    print(' ### INFO: Signal size = \t\t',len(df_sig))
    print(' ### INFO: Background size = \t', len(df_all_bkg))

    ss = sorted(Events['sample'].unique())
    cs = sorted(Events['pairType'].unique())

    print(" ### INFO: Input statistics")
    pt = PrettyTable(['Sample']+[c for c in cs] + ['Tot'])
    for s in ss:
        vs = []
        for c in cs: vs.append(len(Events[(Events['sample'] == s)&(Events['pairType']==c)]))
        vs.append(len(Events[(Events['sample'] == s)]))
        pt.add_row([s]+vs)
    for c in cs: pt.align[c] = "l"
    pt.align['Tot'] = 'l'
    print(pt)

    print(" ### INFO: Input zero weights")
    pt = PrettyTable(['Sample']+[c for c in cs])
    for s in ss:
        vs = []
        for c in cs:
            n = len(Events[(Events['sample'] == s)&(Events['pairType'] == c)&(Events['weight'] == 0)])
            d = len(Events[(Events['sample'] == s)&(Events['pairType'] == c)])
            try: ratio = n/d*100 
            except: ratio = 0
            vs.append("{:.2f}%".format(ratio))
        pt.add_row([s]+vs)
    for c in cs: pt.align[c] = "l"
    print(pt)

    print(" ### INFO: Input negative weights")
    pt = PrettyTable(['Sample']+[c for c in cs])
    for s in ss:
        vs = []
        for c in cs:
            n = len(Events[(Events['sample'] == s)&(Events['pairType'] == c)&(Events['weight'] < 0)])
            d = len(Events[(Events['sample'] == s)&(Events['pairType'] == c)])
            try: ratio = n/d*100 
            except: ratio = 0
            vs.append("{:.2f}%".format(ratio))
        pt.add_row([s]+vs)
    for c in cs: pt.align[c] = "l"
    print(pt)

    ######################### Remove negative weights #########################
    Events = Events[Events['weight'] > 0]

    print(" ### INFO: NaN replacement")
    Events.replace([np.inf, -np.inf], np.nan, inplace=True)
    fix = Events[cont_feat].columns[Events[cont_feat].isna().any()].tolist(); fix

    print(" ### INFO: Replace negative ZZKinFit_chi2 with 0")
    Events.loc[(Events['ZZKinFit_chi2'] < 0), 'ZZKinFit_chi2'] = 0

    ######################### Split even and odd #########################

    print(" ### INFO: Split into even and odd event numbers")
    Events['year'] = 2018
    df_0 = Events[Events['event']%2 == 0] # even event numbers
    df_1 = Events[Events['event']%2 != 0] # odd event numbers

    print(" ### INFO: Save DataFrames")
    odir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/DNNWeights/DNNInputs'
    os.system('mkdir -p ' + odir)
    savepath = Path(odir)
    input_pipe_0 = fit_input_pipe(df_0, cont_feat, savepath/f'input_pipe_0')
    input_pipe_1 = fit_input_pipe(df_1, cont_feat, savepath/f'input_pipe_1')

    set_0_train = df_0.copy()
    set_1_train = df_1.copy()
    set_0_test = df_0.copy()
    set_1_test = df_1.copy()

    # training samples
    set_0_train[cont_feat] = input_pipe_0.transform(set_0_train[cont_feat].values.astype('float32'))
    set_1_train[cont_feat] = input_pipe_1.transform(set_1_train[cont_feat].values.astype('float32'))

    # testing samples
    set_0_test[cont_feat] = input_pipe_1.transform(set_0_test[cont_feat].values.astype('float32'))
    set_1_test[cont_feat] = input_pipe_0.transform(set_1_test[cont_feat].values.astype('float32'))

    ######################### Balance weigths #########################

    set_0_train_weight = balance_weights(set_0_train)
    set_1_train_weight = balance_weights(set_1_train)

    ######################### Plotting inputs #########################

    try:
        a = PlotSettings(w_mid=10, b_mid=10, cat_palette='Set1', style={}, format='pdf')
        b = PlotSettings(w_mid=10, b_mid=10, cat_palette='Set1', style={}, format='png')
        odir = '/eos/user/e/evernazz/www/ZZbbtautau/DNNFeaturePlots/LuminInputs/Inputs0'
        for feature in cont_feat:
            save_name = odir + '/TrainFeat_' + feature
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==0),(set_0_train_weight.Class==1)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==0),(set_0_train_weight.Class==1)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)
        odir = '/eos/user/e/evernazz/www/ZZbbtautau/DNNFeaturePlots/LuminInputs/Inputs1'
        for feature in cont_feat:
            save_name = odir + '/TrainFeat_' + feature
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==0),(set_1_train_weight.Class==1)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==0),(set_1_train_weight.Class==1)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)
    except:
        print(" ### INFO: Skipping plots")

    ######################### Saving #########################

    print(" ### INFO: Saving inputs to fold files")
    df2foldfile(df=set_0_train_weight, n_folds=10,
                cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
                misc_feats=['pairType', 'sample', 'original_weight'],
                savename=savepath/'train_0', targ_type='int')
    
    df2foldfile(df=set_1_train_weight, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            misc_feats=['pairType', 'sample', 'original_weight'],
            savename=savepath/'train_1', targ_type='int')
    
    df2foldfile(df=set_0_test, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            misc_feats=['pairType', 'sample', 'original_weight'],
            savename=savepath/'test_0', targ_type='int')
    
    df2foldfile(df=set_1_test, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            misc_feats=['pairType', 'sample', 'original_weight'],
            savename=savepath/'test_1', targ_type='int')