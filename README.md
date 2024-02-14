# ZZbbtautauDNNInputs

This repository contains scripts for the DNN training of the ZZbbtautau validation analysis.
The DNN training is performed by targeting the ZZ signal sample against all the possible background samples.
The events are selected to enter:
- baseline selections
- Signal Region (OS and Medium working point for tau isolation)
- Elliptical mass cut, here re-optimized to target ZZ/ZH instead of HH

## Prepare environment

```
git clone git@github.com:elenavernazza/ZZbbtautauDNNInputs.git
cd ZZbbtautauDNNInputs
git clone git@github.com:JohanWulff/cms_runII_dnn_resonant.git
```

## Inputs

The inputs are cretaed via the new Framework developed by Jaime: https://github.com/elenavernazza/hhbbtt-analysis
The information about the cross section is read in the config file and saved into the `CrossSection.json` file.

```
python3 GetSamplesConfig.py
```

The ntuples are converted to hdf5 files for the DNN training.

```
python3 ProduceDNNInputs.py --out DNNWeight_M300
```

Inside the script, the events are weighted by the cross section, the generator level weights and the detector related weights. The negative weights are removed.
The event weight is then renormalized in order to give the same importance to the signal sample and to the sum of backgrounds.

## Training

The training needs to be performed on machines with GPU. At LLR, use the llrai machine.
Two trainings have to performed for the even and odd event numbers: they can be run at the same time, selecting the first GPU device for the first training and the second GPU device for the second training (already autmatically selected in the script).

```
python3 TrainDNN.py --num 0
```
```
python3 TrainDNN.py --num 1
```

## Testing

```
python3 TestDNN.py
```

## Export

The export is a delicate process and requires a particular environment. If these instructions are not followed, the exportation still works BUT the output has a format that cannot be used into the default analysis framework (.pb are not files, but folders).
To create the environment:

```
conda env create -f LuminEnv/lumin_3.7.yml
conda activate lumin_3.7

pip install Keras==2.3.0
pip install Keras-Applications==1.0.8
pip install Keras-Preprocessing==1.1.2
pip install numpy==1.21.5
pip install onnx==1.8.0
pip install onnx-tf==1.3.0
pip install protobuf==3.19.4
pip install tensorboard==1.15.0
pip install tensorflow==1.15.0
pip install torch==1.6.0
pip install torchvision==0.7.0
pip install lumin
pip install IPython
pip install pathlib2
```

Once the environment is set it is possible to save the model.

```
python3 SaveDNN.py --name <model_name>
```

python3 GetSamplesConfig.py

python3 ProduceDNNInputs.py --sig ggXZZbbtt_M300 --bkg zz_sl_signal,dy,tt_dl,tt_sl,tt_fh,wjets,ttw_lnu,ttw_qq,ttww,ttwz,ttwh,ttzh,ttz_llnunu,ttz_qq,ttzz,tth_bb,tth_nonbb,tth_tautau,zh_hbb_zll,wplush_htt,wminush_htt,ggH_ZZ,ggf_sm,zz_dl,zz_sl_background,zz_lnu,zz_qnu,wz_lllnu,wz_lnuqq,wz_llqq,wz_lnununu,ww_llnunu,ww_lnuqq,ww_qqqq,zzz,wzz,www,wwz,ewk_z,ewk_wplus,ewk_wminus,st_tw_antitop,st_tw_top,st_antitop,st_top