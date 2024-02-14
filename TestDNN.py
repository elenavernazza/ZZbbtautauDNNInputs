import sys
sys.path.append('../')
from cms_runII_dnn_resonant.modules.data_import import *
from cms_runII_dnn_resonant.modules.basics import *
from cms_runII_dnn_resonant.modules.model_export import *
from cms_runII_dnn_resonant.modules.plotting import *

import copy
from typing import Callable, Tuple
import json
from prettytable import PrettyTable
import math
from sklearn.metrics import roc_auc_score

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

    basedir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/'+options.out+'/'
    weight_dir = basedir + 'ensemble/'

    ensemble_0 = Ensemble.from_save(weight_dir + f'selected_set_0_{run_name}')
    ensemble_1 = Ensemble.from_save(weight_dir + f'selected_set_1_{run_name}')

    indir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/'+options.out+'/DNNInputs'
    inpath = Path(indir)

    set_0_fy = FoldYielder(inpath/'test_0.hdf5', input_pipe=inpath/'input_pipe_0.pkl')
    set_1_fy = FoldYielder(inpath/'test_1.hdf5', input_pipe=inpath/'input_pipe_1.pkl')

    ensemble_0.predict(set_1_fy)
    ensemble_1.predict(set_0_fy)

    df = load_full_df(fy=set_0_fy).append(load_full_df(fy=set_1_fy))

    odir = basedir + '/TestingPerformance/'
    os.system('mkdir -p ' + odir)

    plot_binary_class_pred(df, savename=odir+"Overall")

    print('\nMu Tau')
    plot_binary_class_pred(df[df.channel==1], density=True, savename=odir+"muTau")
    # plot_binary_class_pred(df[df.pairType==0], density=True, savename=odir+"muTau")
    print('E Tau')
    plot_binary_class_pred(df[df.channel==2], density=True, savename=odir+"eTau")
    # plot_binary_class_pred(df[df.pairType==1], density=True, savename=odir+"eTau")
    print('Tau Tau')
    plot_binary_class_pred(df[df.channel==0], density=True, savename=odir+"tauTau")
    # plot_binary_class_pred(df[df.pairType==2], density=True, savename=odir+"tauTau")