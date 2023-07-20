import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(000000)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import mplhep
plt.style.use(mplhep.style.CMS)

def FancyHisto(h, type, num = None):
    h.SetLineWidth(3)
    h.SetTitle("")
    if num == None:
        if type == 'sig':
            h.SetLineColor(ROOT.kRed)
        if type == 'bkg':
            h.SetLineColor(ROOT.kAzure)
    else:
        if num == -1: h.SetLineColor(ROOT.kBlack);    h.SetLineStyle(1)
        if num == 0:  h.SetLineColor(ROOT.kAzure);    h.SetLineStyle(2)
        if num == 1:  h.SetLineColor(ROOT.kGreen);    h.SetLineStyle(3)
        if num == 2:  h.SetLineColor(ROOT.kYellow);   h.SetLineStyle(4)
        if num == 3:  h.SetLineColor(ROOT.kOrange+1); h.SetLineStyle(5)
        if num == 4:  h.SetLineColor(ROOT.kRed);      h.SetLineStyle(6)
        if num == 5:  h.SetLineColor(ROOT.kRed);      h.SetLineStyle(7)
        if num == 6:  h.SetLineColor(ROOT.kPurle);    h.SetLineStyle(8)
        if num == 7:  h.SetLineColor(ROOT.kBlue);     h.SetLineStyle(9)
    return 1

def AddText(leftmargin = 0.10, rightmargin = 0.10, pass1K = False):

    leftshift = leftmargin - 0.10
    if pass1K: leftshift += 0.10
    tex1 = ROOT.TLatex()
    tex1.SetTextSize(0.03)
    tex1.DrawLatexNDC(0.11+leftshift,0.91,"#scale[1.5]{CMS} Private Work")
    tex1.Draw("same")

    rightshift = rightmargin - 0.10
    tex2 = ROOT.TLatex()
    tex2.SetTextSize(0.035)
    tex2.SetTextAlign(31)
    tex2.DrawLatexNDC(0.88-rightshift,0.91,"2018, 13 TeV (59.7 fb^{-1})")
    tex2.Draw("same")

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# To run:
'''
seteos
python3 PlotDNNFeatures.py --sig zz_sl_signal \
 --bkg zz_sl_background,zz_qnu,zz_lnu,zz_dl,zz_fh,zzz,ggf_sm,dy,tt_dl,tt_sl,tt_fh,tth_bb,tth_tautau,tth_nonbb,wjets,st_antitop,st_top,st_tw_antitop,st_tw_top \
 --cat ZZ_elliptical_cut_80_sr --v prod_DNN_Ellipse80_SR
'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--sig",     dest="sig",    default="zz_sl_signal",           help='Signal sample')
    parser.add_option("--bkg",     dest="bkg",    default="wjets,tt_fh,tt_sl,dy",   help='Comma separated list of background samples')
    parser.add_option("--cat",     dest="cat",    default="base_selection",         help='Category name')
    parser.add_option("--ver",     dest="ver",    default="prod_DNN_Ellipse80",     help='Version name')
    parser.add_option("--nev",     dest="nev",    default=100000,                   help='Max number of events')
    (options, args) = parser.parse_args()

    basedir = '/data_CMS/cms/vernazza/cmt/Categorization/ul_2018_ZZ_v10/'
    outdir = '/eos/user/e/evernazz/www/ZZbbtautau/DNNFeaturePlots/'
    L = 59.7 * 1000

    # samples defintion
    sig_dir = basedir + '/' + options.sig + '/cat_' + options.cat + '/' + options.ver
    bkg_list = [options.bkg] if ',' not in options.bkg else options.bkg.split(',')
    bkg_dirs = [basedir + '/' + bkg + '/cat_' + options.cat + '/' + options.ver for bkg in bkg_list]

    # features definition
    cont_feat  = [  'dnn_CvsB_b1', 'dnn_CvsL_b1', 'ZZKinFit_chi2', 'ZZKinFit_mass', 'Ztt_mass',
                    'dnn_dR_l1_l2_x_sv_pT', 'dnn_dau1_mt', 'dau2_pt', 'dnn_dR_l1_l2', 'dnn_dphi_sv_met', 'Zbb_mass',
                    'dnn_HHbtag_b2', 'ZZ_svfit_mass', 'dnn_dphi_Zbb_sv', 'Zbb_pt', 'dnn_dR_l1_l2_boosted_Ztt_met',
                    'dau1_pt', 'dnn_bjet1_pt', 'dnn_Phi', 'dnn_costheta_l2_Zttmet']
    cat_feat   = [  'isBoosted', 'pairType', 'dnn_deepFlav1', 'dnn_deepFlav2', 'VBFjet1_JetIdx']
    weights    = [  'genWeight', 'puWeight', 'prescaleWeight', 'trigSF', 'L1PreFiringWeight_Nom', 'PUjetID_SF']
    feat_names = cont_feat
    n_bins = 100

    features = {
        'dnn_CvsB_b1'                   : {"Min" : 0,   "Max": 1,       "Name": 'CvsB (bJet_{1})'},
        'dnn_CvsL_b1'                   : {"Min" : 0,   "Max": 1,       "Name": 'CvsL (bJet_{1})'},
        'ZZKinFit_chi2'                 : {"Min" : 0,   "Max": 50,      "Name": 'ZZ KinFit #chi^{2}'},
        'ZZKinFit_mass'                 : {"Min" : 0,   "Max": 800,     "Name": 'ZZ KinFit mass'},
        'Ztt_mass'                      : {"Min" : 0,   "Max": 200,     "Name": 'mass (Z_{#tau#tau})'},
        'dnn_dR_l1_l2_x_sv_pT'          : {"Min" : 0,   "Max": 800,     "Name": '#DeltaR (l_{1}, l_{2}) #times p_{T} (Z_{#tau#tau}^{SVFit})'},
        'dnn_dau1_mt'                   : {"Min" : 0,   "Max": 200,     "Name": 'm_{T} (l_{1})'},
        'dau2_pt'                       : {"Min" : 0,   "Max": 200,     "Name": 'p_{T} (l_{2})'},
        'dnn_dR_l1_l2'                  : {"Min" : 0,   "Max": 5,       "Name": '#DeltaR (l_{1}, l_{2})'},
        'dnn_dphi_sv_met'               : {"Min" : 0,   "Max": 3.14,    "Name": '#Delta#phi (Z_{#tau#tau}^{SVFit}, MET)'},
        'Zbb_mass'                      : {"Min" : 0,   "Max": 250,     "Name": 'mass (Z_{bb})'},
        'dnn_HHbtag_b2'                 : {"Min" : 0,   "Max": 1,       "Name": 'ZZbtag bJet_{2}'},
        'ZZ_svfit_mass'                 : {"Min" : 0,   "Max": 800,     "Name": 'mass (ZZ^{SVFit})'},
        'dnn_dphi_Zbb_sv'               : {"Min" : 0,   "Max": 3.14,    "Name": '#Delta#phi (Z_{bb}, Z_{#tau#tau}^{SVFit})'},
        'Zbb_pt'                        : {"Min" : 0,   "Max": 400,     "Name": 'p_{T} (Z_{bb})'},
        'dnn_dR_l1_l2_boosted_Ztt_met'  : {"Min" : 0,   "Max": 5,       "Name": '#DeltaR (l_{1}, l_{2}) Boost(Z_{#tau#tau}+MET)'},
        'dau1_pt'                       : {"Min" : 0,   "Max": 200,     "Name": 'p_{T} (l_{1})'},
        'dnn_bjet1_pt'                  : {"Min" : 0,   "Max": 200,     "Name": 'p_{T} (bJet_{1})'},
        'dnn_Phi'                       : {"Min" : 0,   "Max": 3.14,    "Name": '#Phi'},
        'dnn_costheta_l2_Zttmet'        : {"Min" : 0,   "Max": 1,       "Name": 'cos#Theta'},
    }

    bkg_samples = { 
        'dy':               {'xs': 6077.22,     'nweightedevent': 3323473259844.586},
        'ggf_sm':           {'xs': 0.03105,     'nweightedevent': 400006.41513317823},
        'st_antitop':       {'xs': 80.95,       'nweightedevent': 6114951503.474392},
        'st_top':           {'xs': 136.02,      'nweightedevent': 18955907029.527725},
        'st_tw_antitop':    {'xs': 35.85,       'nweightedevent': 251915355.12840176},
        'st_tw_top':        {'xs': 35.85,       'nweightedevent': 258138394.3995695},
        'tt_dl':            {'xs': 88.29,       'nweightedevent': 10457521197.887497},
        'tt_fh':            {'xs': 377.96,      'nweightedevent': 104893835957.54053},
        'tth_bb':           {'xs': 0.2953,      'nweightedevent': 4843510.130554602},
        'tth_nonbb':        {'xs': 0.17996,     'nweightedevent': 3671542.473303467},
        'tth_tautau':       {'xs': 0.031805,    'nweightedevent': 10899099.713231623},
        'tt_sl':            {'xs': 365.34,      'nweightedevent': 143353855628.27905},
        'wjets':            {'xs': 61526.7,     'nweightedevent': 1190207608.9462342},
        'zz_dl':            {'xs': 1.26,        'nweightedevent': 130482590.99097383},
        'zz_fh':            {'xs': 3.262,       'nweightedevent': 14508674.54780817},
        'zz_lnu':           {'xs': 0.564,       'nweightedevent': 55392591.855656385},
        'zz_qnu':           {'xs': 4.07,        'nweightedevent': 137618971.2015853},
        'zz_sl_background': {'xs': 5.52*0.954,  'nweightedevent': 154801696.57190895},
        'zzz':              {'xs': 0.0147,      'nweightedevent': 3690.177869788371}
    }

    # tree definition
    print(" ### INFO: Signal folder is {}".format(sig_dir))
    sig_tree = ROOT.TChain("Events")
    sig_tree.Add(sig_dir+"/data_*.root")
    sig_nEntries = sig_tree.GetEntries()
    sig_name = options.sig
    sig_xs   = 5.52*0.046
    sig_nev  = 7123786.991464615
    print(" ### INFO: Signal tree has {} entries".format(sig_nEntries))

    h_sig = []
    for feat_name in features.keys():
        min_ = float(features[feat_name]["Min"])
        max_ = float(features[feat_name]["Max"])
        h_sig.append(ROOT.TH1F("h_sig_%s" %feat_name, "h_sig_%s" %feat_name, n_bins, min_, max_))
    
    for i in tqdm(range(0, sig_nEntries)):
        entry = sig_tree.GetEntry(i)
        tot_weight = L * sig_xs / sig_nev * sig_tree.genWeight * sig_tree.puWeight * sig_tree.prescaleWeight * sig_tree.L1PreFiringWeight_Nom
        for j, feat_name in enumerate(features.keys()):
            h_sig[j].Fill(getattr(sig_tree, feat_name), tot_weight)

    h_bkg = []
    for i, bkg_dir in enumerate(bkg_dirs):

        print(" ### INFO: Background folder is {}".format(bkg_dir))
        bkg_tree = ROOT.TChain("Events")
        bkg_tree.Add(bkg_dir+"/data_*.root")
        bkg_nEntries = bkg_tree.GetEntries()
        print(" ### INFO: Background tree has {} entries".format(bkg_nEntries))
        if int(options.nev) == -1: nEvents = bkg_nEntries
        else: nEvents = np.min([bkg_nEntries, int(options.nev)])

        bkg_xs  = bkg_samples[bkg_list[i]]['xs']
        bkg_nev = bkg_samples[bkg_list[i]]['nweightedevent']

        h_bkg_i = []
        for feat_name in features.keys():
            min_ = float(features[feat_name]["Min"])
            max_ = float(features[feat_name]["Max"])
            h_bkg_i.append(ROOT.TH1F("h_bkg%i_%s" %(i,feat_name), "h_bkg%i_%s" %(i,feat_name), n_bins, min_, max_))

        for i in tqdm(range(0, nEvents)):
            entry = bkg_tree.GetEntry(i)
            tot_weight = L * bkg_xs / bkg_nev * bkg_tree.genWeight * bkg_tree.puWeight * bkg_tree.prescaleWeight * bkg_tree.L1PreFiringWeight_Nom
            for j, feat_name in enumerate(features.keys()):
                h_bkg_i[j].Fill(getattr(bkg_tree, feat_name), tot_weight)
        
        h_bkg.append(h_bkg_i)

    # loop on the background
    for i, bkg_name in enumerate(bkg_list):
        # loop on the features
        for j, feat_name in enumerate(features.keys()):

            bkg_out_dir = outdir + '/sig_vs_' + bkg_name
            os.system('mkdir -p ' + bkg_out_dir)
            os.system('cp ' + outdir + '/index.php ' + bkg_out_dir)
            os.system('mkdir -p ' + bkg_out_dir + '/Normalized/')
            os.system('cp ' + outdir + '/index.php ' + bkg_out_dir + '/Normalized/')

            # plots
            c1 = ROOT.TCanvas("c1","c1",800,800)
            c1.SetGrid(10,10)
            c1.SetLeftMargin(0.14)
            if i == 0:
                FancyHisto(h_sig[j], 'sig')
                h_sig[j].GetYaxis().SetTitle("entries")
                h_sig[j].GetXaxis().SetTitle(features[feat_name]['Name'])
            h_sig[j].Draw()
            FancyHisto(h_bkg[i][j], 'bkg')
            h_bkg[i][j].Draw("SAME")
            y_max = np.max([float(h_sig[j].GetMaximum()), float(h_bkg[i][j].GetMaximum())])
            h_sig[j].GetYaxis().SetRangeUser(0,1.1*y_max)

            Legend = ROOT.TLegend(0.65,0.8,0.88,0.88)
            Legend.SetBorderSize(0)
            Legend.AddEntry(h_sig[j] , "Signal", "LPE")
            Legend.AddEntry(h_bkg[i][j] , "Background", "LPE")
            Legend.Draw()

            AddText(leftmargin = 0.14, pass1K = (y_max >= 100))

            c1.SaveAs(bkg_out_dir + '/Feature_%s.png' %(feat_name))
            c1.SaveAs(bkg_out_dir + '/Feature_%s.pdf' %(feat_name))
            del c1

            # normalized plots
            c2 = ROOT.TCanvas("c2","c2",800,800)
            c2.SetGrid(10,10)
            c2.SetLeftMargin(0.14)
            h_sig_norm = h_sig[j]
            h_sig_norm.Scale(1.0 / h_sig[j].Integral())
            h_bkg_norm = h_bkg[i][j]
            h_bkg_norm.Scale(1.0 / h_bkg[i][j].Integral())
            h_sig_norm.GetYaxis().SetTitle("a.u.")
            h_sig_norm.Draw("HIST")
            h_bkg_norm.Draw("HIST,SAME")
            y_max = np.max([float(h_sig_norm.GetMaximum()), float(h_bkg_norm.GetMaximum())])
            h_sig_norm.GetYaxis().SetRangeUser(0,1.1*y_max)

            Legend = ROOT.TLegend(0.65,0.8,0.88,0.88)
            Legend.SetBorderSize(0)
            Legend.AddEntry(h_sig_norm , "Signal", "LPE")
            Legend.AddEntry(h_bkg_norm , bkg_name, "LPE")
            Legend.Draw()

            AddText(leftmargin = 0.14, pass1K = (y_max >= 100))

            c2.SaveAs(bkg_out_dir + '/Normalized/Feature_Norm_%s.png' %(feat_name))
            c2.SaveAs(bkg_out_dir + '/Normalized/Feature_Norm_%s.pdf' %(feat_name))
            del c2, h_sig_norm, h_bkg_norm

    c = ROOT.TCanvas("c","c",800,800)
    c.SetGrid(10,10)
    c.SetLeftMargin(0.14)
    # loop on the background
    # loop on the features
    for j, feat_name in enumerate(features.keys()):

        h_sig_norm = h_sig[j]
        h_sig_norm.Scale(1.0 / h_sig[j].Integral())
        FancyHisto(h_sig_norm, 'sig', num = -1)
        h_sig_norm.GetYaxis().SetTitle("a. u.")
        h_sig_norm.GetXaxis().SetTitle(features[feat_name]['Name'])
        h_sig_norm.Draw("HISTO")

        Legend = ROOT.TLegend(0.65,0.68,0.88,0.88)
        Legend.SetBorderSize(0)
        Legend.AddEntry(h_sig_norm , "Signal", "LPE")
        sig_int = h_sig[j].Integral()

        k = 0
        for i, bkg_name in enumerate(bkg_list):
            if h_bkg[i][j].Integral() < 0.1*sig_int: continue
            print("\n\n ### INFO: Plotting", bkg_name)
            k += 1
            h_bkg_norm = h_bkg[i][j]
            h_bkg_norm.Scale(1.0 / h_bkg[i][j].Integral())
            FancyHisto(h_bkg_norm, 'bkg', num = k)
            h_bkg_norm.Draw("SAME")
            Legend.AddEntry(h_bkg_norm , bkg_name, "LPE")

        Legend.Draw()

        AddText(leftmargin = 0.14, pass1K = (y_max >= 100))

        bkg_out_dir = outdir + '/sig_vs_all'
        os.system('mkdir -p ' + bkg_out_dir)
        os.system('cp ' + outdir + '/index.php ' + bkg_out_dir)
        c.SaveAs(bkg_out_dir + '/Feature_Norm_%s.png' %(feat_name))
        c.SaveAs(bkg_out_dir + '/Feature_Norm_%s.pdf' %(feat_name))
    del c 


    # sig_feat_list = []
    # for i in tqdm(range(0, sig_nEntries)):
    #     entry = sig_tree.GetEntry(i)
    #     sig_feat_list.append(getattr(sig_tree, feat_name))
    # bkg_feat_list = []
    # for i in tqdm(range(0, bkg_nEntries)):
    #     entry = bkg_tree.GetEntry(i)
    #     bkg_feat_list.append(getattr(bkg_tree, feat_name))
    
    # min_ = np.min(sig_feat_list)
    # max_ = np.max(sig_feat_list)
    # binning = np.linspace(min_, max_, n_bins)

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.hist(sig_feat_list, bins=binning, density=1, histtype='step', linewidth=2, label=sig_name)
    # ax.hist(bkg_feat_list, bins=binning, density=1, histtype='step', linewidth=2, label=bkg_name)

    # for xtick in ax.xaxis.get_major_ticks():
    #     xtick.set_pad(10)
    # leg = plt.legend(loc = 'upper right', fontsize=20)
    # leg._legend_box.align = "left"
    # plt.xlabel(feat_name)
    # plt.ylabel('a.u.')
    # for xtick in ax.xaxis.get_major_ticks():
    #     xtick.set_pad(10)
    # plt.grid()
    # mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    # plt.show()
    # plt.savefig('.pdf')
    # plt.close()