import pandas as pd
import numpy as np
import os, sys, glob, json, pdb
import h5py
import pickle
import uproot
import matplotlib
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)
    
from sklearn.model_selection import train_test_split

sys.path.append('../')
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
    pdb.set_trace()
    try:
        tree = root_file[tree_name]
        if len(tree) != 0:
            return tree.arrays(features, library="pd")
    except uproot.exceptions.KeyInFileError:
        print(f" ### ERROR in reading tree '{tree_name}' in the ROOT file: {filename}. Skipping.")

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
                'b_1_cvsb', 'b_1_cvsl', 'boosted_bb', 'boostedTau', 'channel', 'jet_1_quality', 'jet_2_quality', 'year']
weights   = ['genWeightFixed', 'puWeight', 'trigSF', 'PUjetID_SF', 'DYstitchWeight', 
             'L1PreFiringWeight_Nom', 'idAndIsoAndFakeSF', 'bTagweightReshape_smeared']

cont_feat = ['hh_kinfit_chi2', 'hh_kinfit_m', 'sv_mass', 'dR_l1_l2_x_sv_pT', 'l_1_mt', 'l_2_pT', 'dR_l1_l2',
                'dphi_sv_met', 'h_bb_mass', 'b_2_hhbtag', 'diH_mass_sv', 'dphi_hbb_sv', 'h_bb_pT', 
                'dR_l1_l2_boosted_htt_met', 'l_1_pT', 'b_1_pT', 'phi', 'costheta_l2_httmet', 'b_1_cvsb', 'b_1_cvsl']
cat_feat  = ['boosted_bb', 'boostedTau', 'channel', 'jet_1_quality', 'jet_2_quality', 'year']

features = in_feat + weights

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# no environment needed, only seteos

'''
# 2018 only
python3 ProduceDNNInputs.py --out DNNWeight_ZZbbtt_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2018_ZZ_v10 \
 --cat cat_ZZ_elliptical_cut_80_sr --prd prod_240207 --stat_prd prod_240128 --eos True

python3 ProduceDNNInputs.py --out DNNWeight_ZbbHtt_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZbbHtt_v10 \
 --cat cat_ZbbHtt_elliptical_cut_90 --prd prod_240128 --stat_prd prod_240128 --eos True
 
python3 ProduceDNNInputs.py --out DNNWeight_ZttHbb_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZttHbb_v10 \
 --cat cat_ZttHbb_elliptical_cut_90 --prd prod_240128 --stat_prd prod_240128 --eos True

# FullRun2
python3 ProduceDNNInputs.py --out DNNWeight_ZZbbtt_FullRun2_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2016_ZZ_v12,ul_2016_HIPM_ZZ_v12,ul_2017_ZZ_v12,ul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_240318 --stat_prd prod_240305 --eos True

python3 ProduceDNNInputs.py --out DNNWeight_ZbbHtt_FullRun2_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True

python3 ProduceDNNInputs.py --out DNNWeight_ZttHbb_FullRun2_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True

'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    parser.add_option("--sig",       dest="sig",      default='zz_sl_signal')
    parser.add_option("--bkg",       dest="bkg",      default='all')
    parser.add_option("--json",      dest="json",     default='CrossSection.json')
    parser.add_option("--base",      dest="base",     default='/data_CMS/cms/vernazza/cmt/')
    parser.add_option("--ver",       dest="ver",      default='')
    parser.add_option("--cat",       dest="cat",      default='')
    parser.add_option("--prd",       dest="prd",      default='')
    parser.add_option("--stat_prd",  dest="stat_prd", default='')
    parser.add_option("--eos",       dest="eos",      default=None)
    (options, args) = parser.parse_args()

    ######################### Define inputs #########################

    json_path = options.json
    if not os.path.exists(json_path): sys.exit(" ### ERROR: Cross Section json file doesn't exist. Exiting.\n")
    with open(json_path, 'r') as json_file:
        xs_dict = json.load(json_file)

    print("\n ### INFO: Reading signal")
    sig_name = options.sig
    if sig_name in xs_dict.keys():
        print(" - " + sig_name)
    else:
        print(" *** ERROR *** \n", "Dataset not found: {}".format(sig_name))

    print("\n ### INFO: Reading background")
    bkg_names = []
    if options.bkg == 'all': 
        for bkg in xs_dict.keys():
            if bkg == sig_name: continue
            print(" - " + bkg)
            bkg_names.append(bkg)
    else: 
        for bkg in options.bkg.split(','):
            if bkg == sig_name: continue
            if bkg in xs_dict.keys():
                print(" - " + bkg)
                bkg_names.append(bkg)
            else:
                print(" *** ERROR *** \n", "Dataset not found: {}".format(bkg))

    ######################### Define output ########################

    odir = os.getcwd()+'/'+options.out+'/DNNInputs'
    if os.path.isdir(odir):
        print(" ### INFO: Output directory already existing")
    os.system('mkdir -p ' + odir)
    print(" ### INFO: Saving output in", odir)

    ######################### Read inputs #########################

    if ',' in options.ver:
        versions = options.ver.split(',')
        if "ZZ" in options.ver:
            o_name = 'ZZ_FullRun2'
        elif "ZbbHtt" in options.ver:
            o_name = 'ZbbHtt_FullRun2'
        elif "ZttHbb" in options.ver:
            o_name = 'ZttHbb_FullRun2'
    else:
        versions = [options.ver]
        o_name = options.ver

    df_all_sig = pd.DataFrame()
    df_all_bkg = pd.DataFrame()

    for version in versions:

        print('\n ***** INFO ***** : Reading version {}'.format(version))

        data_dir = options.base + '/Categorization/' + version + '/'
        stat_dir = options.base + '/MergeCategorizationStats/' + version + '/'
        print(' ### INFO: Reading datasets in {}'.format(data_dir))
        print(' ### INFO: Reading normalization in {}'.format(stat_dir))

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
        df_sig['gen_weight'] = df_sig['genWeightFixed'] * df_sig['puWeight']
        df_sig['cor_weight'] = df_sig['trigSF'] * df_sig['PUjetID_SF'] * \
                                df_sig['L1PreFiringWeight_Nom'] * df_sig['idAndIsoAndFakeSF'] * \
                                df_sig['DYstitchWeight'] * df_sig['bTagweightReshape_smeared']
        df_sig['weight']     = xs_dict[sig_name]/nweightedevents * df_sig['gen_weight'] * df_sig['cor_weight']
        df_sig['sample']     = sig_name
        df_all_sig = pd.concat([df_all_sig, df_sig], ignore_index=True)
        del data_frames

        for bkg_name in bkg_names:
            files_bkg = glob.glob(data_dir + bkg_name + '/' + options.cat + '/' + options.prd + '/data_*.root')
            print(f" ### INFO: Reading background sample :", data_dir + bkg_name + '/' + options.cat + '/' + options.prd)
            data_frames = [read_root_file(filename, 'Events', features) for filename in files_bkg]
            try:
                df_bkg = pd.concat(data_frames, ignore_index=True)
            except:
                print(" ### ERROR: Empty dataset")
                continue
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
            df_bkg['gen_weight'] = df_bkg['genWeightFixed'] * df_bkg['puWeight']
            df_bkg['cor_weight'] = df_bkg['trigSF'] * df_bkg['PUjetID_SF'] * \
                                   df_bkg['L1PreFiringWeight_Nom'] * df_bkg['idAndIsoAndFakeSF'] * \
                                   df_sig['DYstitchWeight'] * df_sig['bTagweightReshape_smeared']
            if 'dy' not in bkg_name:
                df_bkg['weight']     = xs_dict[bkg_name]/nweightedevents * df_bkg['gen_weight'] * df_bkg['cor_weight']
            else:
                df_bkg['weight']     = xs_dict[bkg_name]/(nweightedevents/nevents) * df_bkg['gen_weight'] * df_bkg['cor_weight']
            df_bkg['sample']     = bkg_name
            df_all_bkg = pd.concat([df_all_bkg, df_bkg], ignore_index=True)
            del data_frames

    df_all_sig.insert(0, 'Class', 1, True)
    df_all_bkg.insert(0, 'Class', 0, True)

    Events = pd.concat([df_all_sig, df_all_bkg])

    ######################### Print statistics #########################

    print(' ### INFO: Signal size = \t',len(df_all_sig))
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
        print(" ### INFO: Plots saved to https://evernazz.web.cern.ch/evernazz/ZZbbtautau/DNNFeaturePlots/" + o_name)
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNFeaturePlots/' + o_name + '/Inputs0'
        os.system('mkdir -p ' + wwwdir)
        os.system('cp /eos/home-e/evernazz/www/index.php /eos/home-e/evernazz/www/ZZbbtautau/DNNFeaturePlots/' + o_name)
        os.system('cp /eos/home-e/evernazz/www/index.php ' + wwwdir)
        for feature in cont_feat:
            save_name = wwwdir + '/TrainFeat_' + feature
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNFeaturePlots/' + o_name + '/Inputs1'
        os.system('mkdir -p ' + wwwdir)
        os.system('cp /eos/home-e/evernazz/www/index.php ' + wwwdir)
        for feature in cont_feat:
            save_name = wwwdir + '/TrainFeat_' + feature
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==1),(set_1_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_1_train_weight, feature, cuts=[(set_1_train_weight.Class==1),(set_1_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)

    if "ZZ" in options.ver:
        pp = 'ZZ'; p_tt = 'Z'; p_bb = 'Z'
    elif "ZbbHtt" in options.ver:
        pp = 'ZH'; p_tt = 'H'; p_bb = 'Z'
    elif "ZttHbb" in options.ver:
        pp = 'ZH'; p_tt = 'Z'; p_bb = 'H'
    cont_feat_name = [  fr'$\chi^{2}$(KinFit)', fr'$M_{{{pp}}}$(KinFit)', fr'$M_{{{p_tt}}}$(SVFit)', fr'$\Delta R (l_{1},l_{2}) \times p_T$(SVFit)', 
                        fr'$m_T (l_{1})$', fr'$p_T (l_{2})$',  fr'$\Delta R (l_{1},l_{2})$', fr'$\Delta\phi (MET, {{{p_tt}}}$(SVFit)$)$',
                        fr'$M ({{{p_bb}}}_{{bb}})$', fr'HHbtag$(b_{2})$', fr'$M_{{{pp}}}$(SVFit)', fr'$\Delta\phi ({{{p_bb}}}_{{bb}}, {{{p_tt}}}$(SVFit)$)$', 
                        fr'$p_T ({{{p_bb}}}_{{bb}})$', fr'$\Delta R (l_{1},l_{2}) \times (MET+{{{p_tt}}}_{{\tau\tau}})$',
                        fr'$p_T (l_{1})$', fr'$p_T (b_{1})$', fr'$\Phi$', fr'$\cos \Theta \,(l_{2}, (MET+{{{p_tt}}}_{{\tau\tau}}))$', fr'CvsB $(b_{1})$', fr'CvsL $(b_{1})$']

    plt.figure(figsize=(40, 35))
    corr_sig = Events[cont_feat].corr()
    heatmap = sns.heatmap(corr_sig, annot=True, xticklabels=cont_feat_name, yticklabels=cont_feat_name, cmap="PiYG", vmin=-1., vmax=1.)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.tight_layout()
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