import sys
from cms_runII_dnn_resonant.modules.data_import import *
from cms_runII_dnn_resonant.modules.basics import *

import torch

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--run",       dest="run",      default='0')
    parser.add_option("--num",       dest="num",      default='0')
    (options, args) = parser.parse_args()

    run_name = options.run
    num = options.num

    print(" ### INFO: Using device {}".format(num))
    torch.cuda.set_device(int(num))
    torch.cuda.get_device_name()

    indir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/DNNWeights/DNNInputs'
    print(" ### INFO: Reading data from {}".format(indir))
    inpath = Path(indir)
    train_fy = FoldYielder(inpath/f'train_{num}.hdf5', input_pipe=inpath/f'input_pipe_{num}.pkl')

    basedir = '/data_CMS/cms/vernazza/FrameworkNanoAOD/DNNTraining/DNNWeights/'
    os.system("mkdir -p " + basedir)
    outdir = basedir + f'train_weights_{num}'
    os.system("mkdir -p " + outdir)
    weightdir = basedir + 'ensemble/'
    os.system("mkdir -p " + weightdir)
    print(" ### INFO: Saving data to {}".format(outdir))
    outpath = Path(outdir)

    bs = 1024 # batch size
    objective = 'classification'
    cat_embedder = CatEmbedder.from_fy(train_fy)
    width = len(train_fy.cont_feats) + np.sum(cat_embedder.emb_szs)

    body = partial(FullyConnected, act='swish', width=width, depth=6, dense=True, do=0.05)
    opt_args = {'opt':'adam', 'eps':1e-08}

    print(" ### INFO: Build model")

    n_out = 1
    model_builder = ModelBuilder(objective, cont_feats=train_fy.cont_feats, n_out=n_out,
                                body=body, opt_args=opt_args)
    Model(model_builder)

    print(" ### INFO: Define learning rate")

    savename = outdir + '/LearningRate'
    lr_finder = lr_find(train_fy, model_builder, bs, lr_bounds=[1e-7,1e1], plot_savename=savename)

    n_models = 10
    patience = 50
    max_epochs = 30

    callback_partials = [partial(OneCycle, lengths=(5, 10), lr_range=[2e-5, 2e-3], mom_range=(0.85, 0.95), interp='cosine')]
    #eval_metrics = [partial(AMS, n_total=2*len(train_fy.get_column('targets')), syst_unc_b=0.1, wgt_name='gen_orig_weight', main_metric=False)]
    eval_metrics = [] 

    print(" ### INFO: Training model with {} models, {} max epochs, patience {}".format(
        n_models, max_epochs, patience))

    results, histories, cycle_losses = train_models(train_fy, n_models,live_fdbk_first_only=False,
                                                       model_builder=model_builder,
                                                       bs=bs,
                                                       cb_partials=callback_partials,
                                                       metric_partials=eval_metrics,
                                                       n_epochs=max_epochs, patience=patience, 
                                                       savepath=outpath)
    
    print(" ### INFO: Ensemble loading")
    with open(outpath/'results_file.pkl', 'rb') as fin:   
        results = pickle.load(fin)

    ensemble = Ensemble.from_results(results, n_models, model_builder, metric='loss', )
    ensemble.add_input_pipe(train_fy.input_pipe)

    name = weightdir + f'selected_set_{num}_{run_name}'
    ensemble.save(name, feats=train_fy.cont_feats + train_fy.cat_feats, overwrite=True)

    savename = outdir + '/Feat_Importance'
    _ = ensemble.get_feat_importance(train_fy, savename=savename)
    ensemble[0][1].head.plot_embeds()