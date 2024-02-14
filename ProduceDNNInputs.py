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
    if len(tree) != 0:
        return tree.arrays(features, library="pd")

def check_weights(df:pd.DataFrame) -> None:
    v = []
    for c in df.Class.unique():
        print('Class ', c)
        for m in df.channel.unique():
            v.append(df.loc[(df.Class==c) & (df.channel==m), 'weight'].sum())
            print(m, 'sum', v[-1])
    print(f' Channel std {np.std(v):.2f}')

def balance_weights(df:pd.DataFrame) -> None:
    df_copy = df.copy()
    print('\n Initial weight sums')
    check_weights(df_copy)
    df_copy['original_weight'] = df_copy['weight']
    for c in df_copy['channel'].unique():
        for t in df_copy['Class'].unique():
            df_copy.loc[(df_copy['Class'] == t) & (df_copy['channel'] == c), 'weight'] \
                /= np.sum(df_copy.loc[(df_copy['Class'] == t) & (df_copy['channel'] == c), 'weight'])
    print('\n Final weight sums')
    check_weights(df_copy)
    return df_copy

in_feat   = ['event', 
                'hh_kinfit_chi2', 'hh_kinfit_m', 'sv_mass', 'dR_l1_l2_x_sv_pT', 'l_1_mt', 'l_2_pT', 'dR_l1_l2',
                'dphi_sv_met', 'h_bb_mass', 'b_2_hhbtag', 'diH_mass_sv', 'dphi_hbb_sv', 'h_bb_pT', 
                'dR_l1_l2_boosted_htt_met', 'l_1_pT', 'b_1_pT', 'phi', 'costheta_l2_httmet', 
                'b_1_cvsb', 'b_1_cvsl', 'boosted', 'channel', 'is_vbf', 'jet_1_quality', 'jet_2_quality', 'year']
weights   = ['genWeight', 'puWeight', 'prescaleWeight', 'trigSF', 'PUjetID_SF']
# weights   = ['genWeight', 'puWeight', 'prescaleWeight', 'trigSF', 'L1PreFiringWeight_Nom', 'PUjetID_SF']

cont_feat = ['hh_kinfit_chi2', 'hh_kinfit_m', 'sv_mass', 'dR_l1_l2_x_sv_pT', 'l_1_mt', 'l_2_pT', 'dR_l1_l2',
                'dphi_sv_met', 'h_bb_mass', 'b_2_hhbtag', 'diH_mass_sv', 'dphi_hbb_sv', 'h_bb_pT', 
                'dR_l1_l2_boosted_htt_met', 'l_1_pT', 'b_1_pT', 'phi', 'costheta_l2_httmet', 'b_1_cvsb', 'b_1_cvsl']
cat_feat  = ['boosted', 'channel', 'is_vbf', 'jet_1_quality', 'jet_2_quality', 'year']

features = in_feat + weights

    # 'zz_sl_background': 5.52*0.954, ########## [FIXME]

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# no environment needed, only seteos

'''
python3 ProduceDNNInputs.py --out DNNWeight_ZZbbtt_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2016_HIPM_ZZ_v10 \
 --cat cat_ZZ_elliptical_cut_80_sr --prd prod_240207 --stat_prd prod_240128 --eos True

python3 ProduceDNNInputs.py --out DNNWeight_ZbbHtt_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZbbHtt_v10 \
 --cat cat_ZbbHtt_elliptical_cut_90 --prd prod_240128 --stat_prd prod_240128 --eos True
 
 python3 ProduceDNNInputs.py --out DNNWeight_ZttHbb_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZttHbb_v10 \
 --cat cat_ZttHbb_elliptical_cut_90 --prd prod_240128 --stat_prd prod_240128 --eos 

/grid_mnt/data__data.polcms/cms/cuisset/cmt/Categorization/ul_2018_ZbbHtt_v10/ewk_wminus/cat_ZbbHtt_elliptical_cut_90/prod_240128
'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    parser.add_option("--sig",       dest="sig",      default='zz_sl_signal')
    parser.add_option("--bkg",       dest="bkg",      default='all')
    parser.add_option("--json",      dest="json",     default='CrossSection.json')
    parser.add_option("--base",      dest="base",     default='/data_CMS/cms/vernazza/cmt/')
    parser.add_option("--ver",       dest="ver",      default='ul_2018_ZZ_v10')
    parser.add_option("--cat",       dest="cat",      default='cat_ZZ_elliptical_cut_80_sr')
    parser.add_option("--prd",       dest="prd",      default='prod_240207')
    parser.add_option("--stat_prd",  dest="stat_prd", default='prod_240128')
    parser.add_option("--eos",       dest="eos",      default=None)
    (options, args) = parser.parse_args()

    ######################### Define inputs #########################

    json_path = options.json
    if not os.path.exists(json_path): sys.exit(" ### ERROR: Cross Section json file doesn't exist. Exiting.\n")
    with open(json_path, 'r') as json_file:
        xs_dict = json.load(json_file)

    sig_name = options.sig
    print(" ### INFO: Signal is", sig_name)
    if not sig_name in xs_dict.keys(): sys.exit(" ### ERROR: Signal Sample not defined in json file. Exiting.\n")
    if options.bkg == 'all': 
        bkg_names = [key for key in xs_dict.keys() if key != sig_name]
    else: 
        bkg_names = [key for key in options.bkg.split(',') if key in xs_dict.keys()]
    print(" ### INFO: Backgrounds are", bkg_names)

    ######################### Define output ########################

    odir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/'+options.out+'/DNNInputs'
    if os.path.isdir(odir):
        print(" ### INFO: Output directory already existing")
        for i in range(0,10):
            odir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/'+options.out+'.{}'.format(i)+'/DNNInputs'
            if not os.path.isdir(odir):
                break
    os.system('mkdir -p ' + odir)
    print(" ### INFO: Saving output in", odir)

    ######################### Read inputs #########################

    # data_dir = '/data_CMS/cms/vernazza/cmt/Categorization/ul_2018_ZZ_v10/'
    # stat_dir = '/data_CMS/cms/vernazza/cmt/MergeCategorizationStats/ul_2018_ZZ_v10/'
    data_dir = options.base + '/Categorization/' + options.ver + '/'
    stat_dir = options.base + '/MergeCategorizationStats/' + options.ver + '/'

    files_sig = glob.glob(data_dir + sig_name + '/' + options.cat + '/' + options.prd + '/data_*.root')
    print(f" ### INFO: Reading signal samples for {sig_name} :", data_dir + sig_name + '/' + options.cat + '/' + options.prd)
    data_frames = [read_root_file(filename, 'Events', features) for filename in files_sig]
    df_sig = pd.concat(data_frames, ignore_index=True)
    stat_file_aux = stat_dir + sig_name + '_aux/' + options.stat_prd + '/stats.json'
    stat_file = stat_dir + sig_name + '/' + options.stat_prd + '/stats.json'
    if os.path.exists(stat_file_aux): stat_file = stat_file_aux
    with open(stat_file, "r") as f: 
        data = json.load(f)
        nevents = data['nevents']
        nweightedevents = data['nweightedevents']
    df_sig['xs']         = xs_dict[sig_name]
    df_sig['nev']        = nevents
    df_sig['nev_w']      = nweightedevents
    df_sig['gen_weight'] = df_sig['genWeight'] * df_sig['puWeight']
    # df_sig['cor_weight'] = df_sig['prescaleWeight'] * df_sig['trigSF'] * df_sig['L1PreFiringWeight_Nom'] * df_sig['PUjetID_SF']
    df_sig['cor_weight'] = df_sig['prescaleWeight'] * df_sig['trigSF'] * df_sig['PUjetID_SF']
    df_sig['weight']     = xs_dict[sig_name]/nweightedevents * df_sig['gen_weight'] * df_sig['cor_weight']
    df_sig['sample']     = sig_name
    del data_frames

    df_all_bkg = pd.DataFrame()
    for bkg_name in bkg_names:
        files_bkg = glob.glob(data_dir + bkg_name + '/' + options.cat + '/' + options.prd + '/data_*.root')
        print(" ### INFO: Reading background samples for", bkg_name)
        data_frames = [read_root_file(filename, 'Events', features) for filename in files_bkg]
        df_bkg = pd.concat(data_frames, ignore_index=True)
        if bkg_name != 'dy':
            stat_file_aux = stat_dir + bkg_name + '_aux/' + options.stat_prd + '/stats.json'
        else:
            stat_dir + 'dy_nlo_aux/' + options.stat_prd + '/stats.json'
        stat_file = stat_dir + bkg_name + '/' + options.stat_prd + '/stats.json'
        if os.path.exists(stat_file_aux): stat_file = stat_file_aux
        with open(stat_file, "r") as f: 
            data = json.load(f)
            nevents = data['nevents']
            nweightedevents = data['nweightedevents']
        df_bkg['xs']         = xs_dict[bkg_name]
        df_bkg['nev']        = nevents
        df_bkg['nev_w']      = nweightedevents
        df_bkg['gen_weight'] = df_bkg['genWeight'] * df_bkg['puWeight']
        # df_bkg['cor_weight'] = df_bkg['prescaleWeight'] * df_bkg['trigSF'] * df_bkg['L1PreFiringWeight_Nom'] * df_bkg['PUjetID_SF']
        df_bkg['cor_weight'] = df_bkg['prescaleWeight'] * df_bkg['trigSF'] * df_bkg['PUjetID_SF']
        df_bkg['weight']     = xs_dict[bkg_name]/nweightedevents * df_bkg['gen_weight'] * df_bkg['cor_weight']
        df_bkg['sample']     = bkg_name
        print(" ### INFO: Appending")
        df_all_bkg = pd.concat([df_all_bkg, df_bkg], ignore_index=True)
        del data_frames

    df_sig.insert(0, 'Class', 1, True)
    df_all_bkg.insert(0, 'Class', 0, True)

    Events = pd.concat([df_sig, df_all_bkg])

    ######################### Print statistics #########################

    print(' ### INFO: Signal size = \t',len(df_sig))
    print(' ### INFO: Background size = \t', len(df_all_bkg))

    ss = sorted(Events['sample'].unique())
    cs = sorted(Events['channel'].unique())

    print(" ### INFO: Input statistics")
    pt = PrettyTable(['Sample']+[c for c in cs] + ['Tot'])
    for s in ss:
        vs = []
        for c in cs: vs.append(len(Events[(Events['sample'] == s)&(Events['channel']==c)]))
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
            n = len(Events[(Events['sample'] == s)&(Events['channel'] == c)&(Events['weight'] == 0)])
            d = len(Events[(Events['sample'] == s)&(Events['channel'] == c)])
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
            n = len(Events[(Events['sample'] == s)&(Events['channel'] == c)&(Events['weight'] < 0)])
            d = len(Events[(Events['sample'] == s)&(Events['channel'] == c)])
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

    # print(" ### INFO: Replace negative ZZKinFit_chi2 with 0")
    # Events.loc[(Events['ZZKinFit_chi2'] < 0), 'ZZKinFit_chi2'] = 0

    ######################### Split even and odd #########################

    print(" ### INFO: Split into even and odd event numbers")
    df_0 = Events[Events['event']%2 == 0] # even event numbers
    df_1 = Events[Events['event']%2 != 0] # odd event numbers

    print(" ### INFO: Save DataFrames")
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

    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    a = PlotSettings(w_mid=10, b_mid=10, cat_palette='Set1', style={}, format='pdf')
    b = PlotSettings(w_mid=10, b_mid=10, cat_palette='Set1', style={}, format='png')

    if options.eos != None:
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNFeaturePlots/' + options.ver + '/Inputs0'
        os.system('mkdir -p ' + wwwdir)
        for feature in cont_feat:
            save_name = wwwdir + '/TrainFeat_' + feature
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNFeaturePlots/' + options.ver + '/Inputs1'
        os.system('mkdir -p ' + wwwdir)
        for feature in cont_feat:
            save_name = wwwdir + '/TrainFeat_' + feature
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==1),(set_1_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==1),(set_1_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)

    plt.figure(figsize=(40, 35))
    corr_sig = Events[cont_feat].corr()
    sns.heatmap(corr_sig, annot=True, xticklabels=cont_feat, yticklabels=cont_feat, cmap="PiYG")
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.savefig(odir + '/TrainFeat_CorrMatrix.pdf')

    ######################### Saving #########################

    print(" ### INFO: Saving inputs to fold files")
    df2foldfile(df=set_0_train_weight, n_folds=10,
                cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
                savename=savepath/'train_0', targ_type='int')
    
    df2foldfile(df=set_1_train_weight, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            savename=savepath/'train_1', targ_type='int')
    
    df2foldfile(df=set_0_test, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            savename=savepath/'test_0', targ_type='int')
    
    df2foldfile(df=set_1_test, n_folds=10,
            cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
            savename=savepath/'test_1', targ_type='int')