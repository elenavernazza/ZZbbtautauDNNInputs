import pandas as pd
import numpy as np
import os, sys, glob, json, pdb, random
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
    try:
        tree = root_file[tree_name]
        if len(tree) != 0:
            return tree.arrays(features, library="pd")
    except uproot.exceptions.KeyInFileError:
        print(f"'{tree_name}' does not exist in the ROOT file: {filename}.")

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

cont_feat = ['mass', 'hh_kinfit_chi2', 'hh_kinfit_m', 'sv_mass', 'dR_l1_l2_x_sv_pT', 'l_1_mt', 'l_2_pT', 'dR_l1_l2',
                'dphi_sv_met', 'h_bb_mass', 'b_2_hhbtag', 'diH_mass_sv', 'dphi_hbb_sv', 'h_bb_pT', 
                'dR_l1_l2_boosted_htt_met', 'l_1_pT', 'b_1_pT', 'phi', 'costheta_l2_httmet', 'b_1_cvsb', 'b_1_cvsl']
cat_feat  = ['boosted', 'channel', 'is_vbf', 'jet_1_quality', 'jet_2_quality', 'year']

features = in_feat + weights

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# no environment needed, only seteos

'''
# 2018 ZZ first test
python3 ProduceDNNInputs.py --out ResTest1 \
 --sig ggXZZbbtt_M200,ggXZZbbtt_M300,ggXZZbbtt_M400,ggXZZbbtt_M500,ggXZZbbtt_M600,ggXZZbbtt_M700,ggXZZbbtt_M800,ggXZZbbtt_M900,\
ggXZZbbtt_M1000,ggXZZbbtt_M1100,ggXZZbbtt_M1200,ggXZZbbtt_M1300,ggXZZbbtt_M1400,ggXZZbbtt_M1500,ggXZZbbtt_M2000,ggXZZbbtt_M3000 \
 --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2018_ZZ_v10 \
 --cat cat_ZZ_elliptical_cut_80_sr --prd prod_240207 --stat_prd prod_231005 --eos True

python3 ProduceDNNInputs.py --out ResTest2 \
 --sig ggXZZbbtt_M200,ggXZZbbtt_M300 \
 --bkg www --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2018_ZZ_v10 \
 --cat cat_ZZ_elliptical_cut_80_sr --prd prod_240207 --stat_prd prod_231005 --eos False

python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZbbHtt_0 \
 --sig Zprime_Zh_Zbbhtautau_M500_v3,Zprime_Zh_Zbbhtautau_M1000_v3,Zprime_Zh_Zbbhtautau_M2000_v3,Zprime_Zh_Zbbhtautau_M3000_v3,Zprime_Zh_Zbbhtautau_M4000_v3 \
 --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True

python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZttHbb_0 \
 --sig Zprime_Zh_Ztautauhbb_M500_v3,Zprime_Zh_Ztautauhbb_M1000_v3,Zprime_Zh_Ztautauhbb_M2000_v3,Zprime_Zh_Ztautauhbb_M3000_v3,Zprime_Zh_Ztautauhbb_M4000_v3 \
 --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True

'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    parser.add_option("--sig",       dest="sig",      default='ggXZZbbtt_M200')
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
    sig_names = []
    if ',' in options.sig:
        for sig in options.sig.split(','):
            if sig in xs_dict.keys():
                print(" - " + sig)
                sig_names.append(sig)
            else:
                print(" *** ERROR *** \n", "Dataset not found: {}".format(sig))
    else:
        print(" - " + options.sig)
        sig_names = [options.sig]

    print("\n ### INFO: Reading background")
    bkg_names = []
    if options.bkg == 'all': 
        for bkg in xs_dict.keys():
            if bkg in sig_names: continue
            print(" - " + bkg)
            bkg_names.append(bkg)
    else: 
        for bkg in options.bkg.split(','):
            if bkg in sig_names: continue
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

        mass_points = []

        for sig_name in sig_names:
            files_sig = glob.glob(data_dir + sig_name + '/' + options.cat + '/' + options.prd + '/data_*.root')
            if "_v3" in sig_name: mass = float(sig_name.split("M")[-1].split("_v3")[0])
            else: mass = float(sig_name.split("M")[-1])
            print(f" ### INFO: Reading signal samples for mass {mass} GeV :", data_dir + sig_name + '/' + options.cat + '/' + options.prd)
            data_frames = [read_root_file(filename, 'Events', features) for filename in files_sig]
            if all(element is None for element in data_frames): 
                print(" ### EMPTY DATASET")
                continue
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
            df_sig['cor_weight'] = df_sig['prescaleWeight'] * df_sig['trigSF'] * df_sig['PUjetID_SF'] # [FIXME] Add DYStitching
            df_sig['weight']     = xs_dict[sig_name]/nweightedevents * df_sig['gen_weight'] * df_sig['cor_weight']
            df_sig['sample']     = sig_name
            df_sig['mass']       = mass
            df_all_sig = pd.concat([df_all_sig, df_sig], ignore_index=True)
            mass_points.append(mass)
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
            df_bkg['gen_weight'] = df_bkg['genWeight'] * df_bkg['puWeight']
            df_bkg['cor_weight'] = df_bkg['prescaleWeight'] * df_bkg['trigSF'] * df_bkg['PUjetID_SF'] # * df_bkg['DYstitchEasyWeight']
            df_bkg['weight']     = xs_dict[bkg_name]/nweightedevents * df_bkg['gen_weight'] * df_bkg['cor_weight']
            df_bkg['sample']     = bkg_name
            df_bkg['mass']       = random.choices(mass_points, k=len(df_bkg))
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

    # save one testing sample for each mass value
    set_0_test_vector = []
    set_1_test_vector = []
    for sig_name in sig_names:
        df_0_i_mass = df_0.copy()
        df_1_i_mass = df_1.copy()
        # drop all the other mass points and only keep the corresponding mass point for the signal
        for sig_to_remove in sig_names:
            if sig_to_remove == sig_name: continue
            df_0_i_mass.drop(df_0_i_mass[df_0_i_mass['sample'] == sig_to_remove].index, inplace=True)
            df_1_i_mass.drop(df_1_i_mass[df_1_i_mass['sample'] == sig_to_remove].index, inplace=True)
        # set the background mass to that mass point
        if "_v3" in sig_name: i_mass = float(sig_name.split("M")[-1].split("_v3")[0])
        else: i_mass = float(sig_name.split("M")[-1])
        df_0_i_mass['mass'] = float(i_mass)
        df_1_i_mass['mass'] = float(i_mass)

        df_0_i_mass[cont_feat] = input_pipe_1.transform(df_0_i_mass[cont_feat].values.astype('float32'))
        df_1_i_mass[cont_feat] = input_pipe_0.transform(df_1_i_mass[cont_feat].values.astype('float32'))

        set_0_test_vector.append(df_0_i_mass)
        set_1_test_vector.append(df_1_i_mass)

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
        print(" ### INFO: Plots saved to https://evernazz.web.cern.ch/evernazz/ZZbbtautau/DNNResFeaturePlots/" + o_name)
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNResFeaturePlots/' + o_name + '/Inputs0'
        os.system('mkdir -p ' + wwwdir)
        os.system('cp /eos/home-e/evernazz/www/index.php /eos/home-e/evernazz/www/ZZbbtautau/DNNResFeaturePlots/' + o_name)
        os.system('cp /eos/home-e/evernazz/www/index.php ' + wwwdir)
        for feature in cont_feat:
            if feature == "mass": continue
            save_name = wwwdir + '/TrainFeat_' + feature
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=a)
            plot_feat(set_0_train_weight, feature, cuts=[(set_0_train_weight.Class==1),(set_0_train_weight.Class==0)], labels=['Sig','Bkg'], wgt_name='weight', savename=save_name, settings=b)
        wwwdir = '/eos/home-e/evernazz/www/ZZbbtautau/DNNResFeaturePlots/' + options.ver + '/Inputs1'
        os.system('mkdir -p ' + wwwdir)
        os.system('cp /eos/home-e/evernazz/www/index.php ' + wwwdir)
        for feature in cont_feat:
            if feature == "mass": continue
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

    for i, sig_name in enumerate(sig_names):
        if '_v3' in sig_name:
            output_name = "M" + sig_name.split("_M")[1].split("_v3")[0]
        else:
            output_name = "M" + sig_name.split("_M")[1]
        df2foldfile(df=set_0_test_vector[i], n_folds=10,
                cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
                savename=savepath/f'test_{output_name}_0', targ_type='int')
        df2foldfile(df=set_1_test_vector[i], n_folds=10,
                cont_feats=cont_feat, cat_feats=cat_feat, targ_feats='Class', wgt_feat='weight',
                savename=savepath/f'test_{output_name}_1', targ_type='int')