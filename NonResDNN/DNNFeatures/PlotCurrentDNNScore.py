import ROOT
import matplotlib.pyplot as plt
import numpy as np
import mplhep
plt.style.use(mplhep.style.CMS)

sigBranch = 'zz_sl_signal'
bkgBranch = 'background'

qcd = True

for ch in ['etau', 'mutau', 'tautau']:
    indir = f'/grid_mnt/data__data.polcms/cms/vernazza/cmt/FeaturePlot/ul_2018_ZZ_v10/cat_ZZ_elliptical_cut_80_{ch}/prod_230718_OldDNNScore/'
    if not qcd: file_name = indir + f'root/dnn_zzbbtt_kl_1__{ch}_os_iso__pg_zz_total__nodata.root'
    else: file_name = indir + f'root/dnn_zzbbtt_kl_1__{ch}_os_iso__pg_zz_total_data__qcd__nodata.root'
    file_in  = ROOT.TFile(file_name, 'r')
    sig = file_in.Get('histograms/'+sigBranch)
    bkg = file_in.Get('histograms/'+bkgBranch)

    fig, ax = plt.subplots(figsize=(10,10))

    X = [] ; Y = [] ; X_err = [] ; Y_err = []
    for ibin in range(0,sig.GetNbinsX()):
        X.append(sig.GetBinLowEdge(ibin+1) + sig.GetBinWidth(ibin+1)/2.)
        Y.append(sig.GetBinContent(ibin+1)/np.sqrt(bkg.GetBinContent(ibin+1)))
        X_err.append(sig.GetBinWidth(ibin+1)/2.)
        Y_err.append(sig.GetBinError(ibin+1)/np.sqrt(bkg.GetBinContent(ibin+1)))
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, color='black', zorder=0)

    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    leg = plt.legend(loc='upper right', fontsize=20)
    leg._legend_box.align = "left"
    plt.xlabel("DNN Score")
    plt.ylabel(r'$S/\sqrt{B}$')
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    plt.grid()
    mplhep.cms.label(data=False, rlabel='(13 TeV)')
    if not qcd: savename = f'./SoverB_{ch}'
    else: savename = f'./SoverB_{ch}_qcd'
    plt.savefig(savename+'.png')
    plt.savefig(savename+'.pdf')
    plt.close()