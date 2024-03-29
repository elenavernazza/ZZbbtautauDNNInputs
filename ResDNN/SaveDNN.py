import numpy as np
import sys, os
import torch

from lumin.nn.ensemble.ensemble import Ensemble
from lumin.nn.models.model import Model
from pathlib2 import Path

from typing import Union
def write_ensemble_file(ensemble:Ensemble, basic_name:str, savename:Union[str,Path]) -> None:
    with open(savename, 'w') as fout:
        for i, w in enumerate(ensemble.weights): fout.write(f'{basic_name}_{i} {w}\n')

def write_preproc_file(scalar, savename:Union[str,Path]) -> None:
    with open(savename, 'w') as fout:
        for m,s in zip(scalar.mean_,scalar.scale_): fout.write(f'{m} {s}\n')

def write_feat_file(feats, savename:Union[str,Path]) -> None:
    with open(savename, 'w') as fout:
        for f in feats: fout.write(f'{f}\n')

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# To run:
# conda activate lumin_3.7

'''
python3 SaveDNN.py --out DNNWeight_ZZbbtt_0 --run 0 --name ZZbbtt

python3 SaveDNN.py --out DNNWeight_ZbbHtt_0 --run 0 --name ZbbHtt
 
python3 SaveDNN.py --out DNNWeight_ZttHbb_0 --run 0 --name ZttHbb
'''

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",       dest="run",      default='0')
    parser.add_option("--name",      dest="name",     default='model')
    parser.add_option("--out",       dest="out",      default='DNNWeight_0')
    parser.add_option("--fold",      dest="fold",     default=None)
    (options, args) = parser.parse_args()

    run_name = options.run
    ch_name = 'zz_bbtt'

    basedir = os.getcwd()+'/'+options.out+'/'
    weight_dir = basedir + 'ensemble/'

    print(" ### INFO: Import models")
    ensemble_0 = Ensemble.from_save(weight_dir + f'/selected_set_0_{run_name}')
    ensemble_1 = Ensemble.from_save(weight_dir + f'/selected_set_1_{run_name}')

    modeldir = basedir + '/' + options.name + '-' + run_name
    os.system('mkdir -p ' + modeldir)
    
    if not os.path.exists(modeldir + '/ensemble_0/'):
        os.makedirs(modeldir + '/ensemble_0/')
    if not os.path.exists(modeldir + '/ensemble_1/'):
        os.makedirs(modeldir + '/ensemble_1/')

    print(" ### INFO: Export model")
    ensemble_0.export2tfpb(modeldir + f'/ensemble_0/{ch_name}')
    ensemble_0.export2onnx(modeldir + f'/ensemble_0/{ch_name}')
    ensemble_1.export2tfpb(modeldir + f'/ensemble_1/{ch_name}')
    ensemble_1.export2onnx(modeldir + f'/ensemble_1/{ch_name}')

    print(" ### INFO: Write weights")
    write_ensemble_file(ensemble_0, ch_name, modeldir + '/ensemble_0/model_weights.txt')
    write_ensemble_file(ensemble_1, ch_name, modeldir + '/ensemble_1/model_weights.txt')

    from lumin.nn.data.fold_yielder import FoldYielder
    inpath = Path(os.getcwd()+'/'+options.out+'/DNNInputs')

    train_0_fy = FoldYielder(inpath/'train_0.hdf5', input_pipe=f'{inpath}/input_pipe_0.pkl')
    train_1_fy = FoldYielder(inpath/'train_1.hdf5', input_pipe=f'{inpath}/input_pipe_1.pkl')

    write_preproc_file(train_0_fy.input_pipe['norm_in'], modeldir+'/ensemble_0/preproc.txt')
    write_preproc_file(train_1_fy.input_pipe['norm_in'], modeldir+'/ensemble_1/preproc.txt')

    cat_feats  = train_0_fy.get_use_cat_feats()
    cont_feats = train_0_fy.get_use_cont_feats()
    feats = cont_feats + cat_feats

    write_feat_file(feats, modeldir+'/features.txt')

    bbtt = '/grid_mnt/data__data.polcms/cms/vernazza/FrameworkNanoAOD/hhbbtt-analysis/'
    outdir = bbtt + 'nanoaod_base_analysis/data/cmssw/CMSSW_12_3_0_pre6/src/cms_runII_dnn_models/models/arc_checks/zz_bbtt/'
    if options.fold != None:
        outdir = outdir + '/' + options.fold
        os.system(f'mkdir -p {outdir}')
    
    print(f' ### INFO: From {modeldir}')
    print(f' ### INFO: To {outdir}')
    answer = input(" ### INFO: Do you want to copy the model? [y/n]")
    if answer == 'y':
        os.system('cp -r ' + modeldir + ' ' + outdir)

    print(" ### INFO: Done!")