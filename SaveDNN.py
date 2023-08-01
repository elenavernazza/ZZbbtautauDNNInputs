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

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",       dest="run",      default='0')
    parser.add_option("--name",      dest="name",     default='model')
    (options, args) = parser.parse_args()

    run_name = options.run
    ch_name = 'zz_bbtt'

    basedir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/DNNWeightsDefault/'
    weight_dir = basedir + 'ensemble/'

    print(" ### INFO: Import models")
    ensemble_0 = Ensemble.from_save(weight_dir + f'/selected_set_0_{run_name}')
    ensemble_1 = Ensemble.from_save(weight_dir + f'/selected_set_1_{run_name}')

    modeldir = basedir + '/' + options.name + '_' + run_name
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
    inpath = Path('/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/DNNWeightsDefault/DNNInputs')

    train_0_fy = FoldYielder(inpath/'train_0.hdf5', input_pipe=f'{inpath}/input_pipe_0.pkl')
    train_1_fy = FoldYielder(inpath/'train_1.hdf5', input_pipe=f'{inpath}/input_pipe_1.pkl')

    write_preproc_file(train_0_fy.input_pipe['norm_in'], modeldir+'/ensemble_0/preproc.txt')
    write_preproc_file(train_1_fy.input_pipe['norm_in'], modeldir+'/ensemble_1/preproc.txt')

    cat_feats  = train_0_fy.get_use_cat_feats()
    cont_feats = train_0_fy.get_use_cont_feats()
    feats = cont_feats + cat_feats

    write_feat_file(feats, modeldir+'/features.txt')

    print(" ### INFO: Done!")