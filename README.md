# ZZbbtautauDNNInputs

This repository contains scripts for the DNN training of the ZZbbtautau validation analysis.
The DNN training is performed by targeting the ZZ signal sample against all the possible background samples.
The events are selected to enter:
- baseline selections
- Signal Region (OS and Medium working point for tau isolation)
- Elliptical mass cut, here re-optimized to target ZZ/ZH instead of HH

## Prepare environment

```bash
git clone git@github.com:elenavernazza/ZZbbtautauDNNInputs.git
cd ZZbbtautauDNNInputs
git clone git@github.com:JohanWulff/cms_runII_dnn_resonant.git
```

For the Non-Resonant training, choose `NonResDNN`:
```bash
cd NonResDNN
```

<details>
<summary>FullRun2 NonResDNN commands with Boosted Taus</summary>

```bash

python3 ProduceDNNInputs.py --out 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZZ_v12,bul_2016_HIPM_ZZ_v12,bul_2017_ZZ_v12,bul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZZbbtt.log
python3 ProduceDNNInputs.py --out 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZbbHtt_v12,bul_2016_HIPM_ZbbHtt_v12,bul_2017_ZbbHtt_v12,bul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_OC90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZbbHtt.log
python3 ProduceDNNInputs.py --out 2025_03_04/DNNWeight_ZttHbb_FullRun2_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZttHbb_v12,bul_2016_HIPM_ZttHbb_v12,bul_2017_ZttHbb_v12,bul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_OC90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZttHbb.log
```

```bash
python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --num 0 | tee 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0/training_0.log
python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --num 1 | tee 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0/training_1.log

python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --num 0 | tee 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0/training_0.log
python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --num 1 | tee 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0/training_1.log
 
python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZttHbb_FullRun2_0 --run 0 --num 0 | tee 2025_03_04/DNNWeight_ZttHbb_FullRun2_0/training_0.log
python3 TrainDNN.py --out 2025_03_04/DNNWeight_ZttHbb_FullRun2_0 --run 0 --num 1 | tee 2025_03_04/DNNWeight_ZttHbb_FullRun2_0/training_1.log
```

```bash
python3 TestDNN.py --out 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0 --run 0 
python3 TestDNN.py --out 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0 --run 0
python3 TestDNN.py --out 2025_03_04/DNNWeight_ZttHbb_FullRun2_0 --run 0
```

```bash
conda activate lumin_3.7
python3 SaveDNN.py --out 2025_03_04/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --name ZZbbtt --fold 2025-01-09
python3 SaveDNN.py --out 2025_03_04/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --name ZbbHtt --fold 2025-01-09
python3 SaveDNN.py --out 2025_03_04/DNNWeight_ZttHbb_FullRun2_0 --run 0 --name ZttHbb --fold 2025-01-09
```


</details>

<details>
<summary>FullRun2 NonResDNN commands</summary>

```bash
python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZZbbtt_FullRun2_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2016_ZZ_v12,ul_2016_HIPM_ZZ_v12,ul_2017_ZZ_v12,ul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_240318 --stat_prd prod_240305 --eos True
python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZbbHtt_FullRun2_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True
python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZttHbb_FullRun2_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True
```

```bash
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --num 1

python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --num 1
 
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZttHbb_FullRun2_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZttHbb_FullRun2_0 --run 0 --num 1
```

```bash
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_FullRun2_0 --run 0 
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_FullRun2_0 --run 0
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZttHbb_FullRun2_0 --run 0
```

```bash
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_FullRun2_0 --run 0 --name ZZbbtt --fold 2024-03-26
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_FullRun2_0 --run 0 --name ZbbHtt --fold 2024-03-26
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZttHbb_FullRun2_0 --run 0 --name ZttHbb --fold 2024-03-26
```

</details>

For the Resonant (parametrized) training, choose `ResDNN`:
```bash
cd ResDNN
```

<details>
<summary>FullRun2 ResDNN commands with Boosted Taus</summary>

```bash
python3 -u ProduceDNNInputs.py --out 2025_01_09/DNNWeight_ZZbbtt_0 \
 --sig GluGluToXToZZTo2B2Tau_M200,GluGluToXToZZTo2B2Tau_M210,GluGluToXToZZTo2B2Tau_M220,GluGluToXToZZTo2B2Tau_M230,GluGluToXToZZTo2B2Tau_M240,GluGluToXToZZTo2B2Tau_M250,GluGluToXToZZTo2B2Tau_M260,GluGluToXToZZTo2B2Tau_M270,GluGluToXToZZTo2B2Tau_M280,GluGluToXToZZTo2B2Tau_M300,GluGluToXToZZTo2B2Tau_M320,GluGluToXToZZTo2B2Tau_M350,GluGluToXToZZTo2B2Tau_M360,GluGluToXToZZTo2B2Tau_M400,GluGluToXToZZTo2B2Tau_M450,GluGluToXToZZTo2B2Tau_M500,GluGluToXToZZTo2B2Tau_M550,GluGluToXToZZTo2B2Tau_M600,GluGluToXToZZTo2B2Tau_M650,GluGluToXToZZTo2B2Tau_M700,GluGluToXToZZTo2B2Tau_M750,GluGluToXToZZTo2B2Tau_M800,GluGluToXToZZTo2B2Tau_M850,GluGluToXToZZTo2B2Tau_M900,GluGluToXToZZTo2B2Tau_M1000,GluGluToXToZZTo2B2Tau_M1100,GluGluToXToZZTo2B2Tau_M1200,GluGluToXToZZTo2B2Tau_M1300,GluGluToXToZZTo2B2Tau_M1400,GluGluToXToZZTo2B2Tau_M1500,GluGluToXToZZTo2B2Tau_M1600,GluGluToXToZZTo2B2Tau_M1700,GluGluToXToZZTo2B2Tau_M1800,GluGluToXToZZTo2B2Tau_M1900,GluGluToXToZZTo2B2Tau_M2000,GluGluToXToZZTo2B2Tau_M2200,GluGluToXToZZTo2B2Tau_M2400,GluGluToXToZZTo2B2Tau_M2500,GluGluToXToZZTo2B2Tau_M2600,GluGluToXToZZTo2B2Tau_M2800,GluGluToXToZZTo2B2Tau_M3000,GluGluToXToZZTo2B2Tau_M3500,GluGluToXToZZTo2B2Tau_M4000,GluGluToXToZZTo2B2Tau_M4500,GluGluToXToZZTo2B2Tau_M5000 \
 --bkg all --json CrossSectionZZ.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZZ_v12,bul_2016_HIPM_ZZ_v12,bul_2017_ZZ_v12,bul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZZbbtt.log

python3 -u ProduceDNNInputs.py --out 2025_01_09/DNNWeight_ZbbHtt_0 \
 --sig ZprimeToZH_ZToBB_HToTauTau_M500,ZprimeToZH_ZToBB_HToTauTau_M600,ZprimeToZH_ZToBB_HToTauTau_M700,ZprimeToZH_ZToBB_HToTauTau_M800,ZprimeToZH_ZToBB_HToTauTau_M1000,ZprimeToZH_ZToBB_HToTauTau_M1200,ZprimeToZH_ZToBB_HToTauTau_M1400,ZprimeToZH_ZToBB_HToTauTau_M1600,ZprimeToZH_ZToBB_HToTauTau_M1800,ZprimeToZH_ZToBB_HToTauTau_M2000,ZprimeToZH_ZToBB_HToTauTau_M2500,ZprimeToZH_ZToBB_HToTauTau_M3000,ZprimeToZH_ZToBB_HToTauTau_M3500,ZprimeToZH_ZToBB_HToTauTau_M4000,ZprimeToZH_ZToBB_HToTauTau_M4500,ZprimeToZH_ZToBB_HToTauTau_M5000,ZprimeToZH_ZToBB_HToTauTau_M5500,ZprimeToZH_ZToBB_HToTauTau_M6000 \
 --bkg all --json CrossSectionZbbHtt.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZbbHtt_v12,bul_2016_HIPM_ZbbHtt_v12,bul_2017_ZbbHtt_v12,bul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_OC90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZbbHtt.log

python3 -u ProduceDNNInputs.py --out 2025_01_09/DNNWeight_ZttHbb_0 \
 --sig ZprimeToZH_ZToTauTau_HToBB_M500,ZprimeToZH_ZToTauTau_HToBB_M600,ZprimeToZH_ZToTauTau_HToBB_M700,ZprimeToZH_ZToTauTau_HToBB_M800,ZprimeToZH_ZToTauTau_HToBB_M1000,ZprimeToZH_ZToTauTau_HToBB_M1200,ZprimeToZH_ZToTauTau_HToBB_M1400,ZprimeToZH_ZToTauTau_HToBB_M1600,ZprimeToZH_ZToTauTau_HToBB_M1800,ZprimeToZH_ZToTauTau_HToBB_M2000,ZprimeToZH_ZToTauTau_HToBB_M2500,ZprimeToZH_ZToTauTau_HToBB_M3000,ZprimeToZH_ZToTauTau_HToBB_M3500,ZprimeToZH_ZToTauTau_HToBB_M4000,ZprimeToZH_ZToTauTau_HToBB_M4500,ZprimeToZH_ZToTauTau_HToBB_M5000,ZprimeToZH_ZToTauTau_HToBB_M5500,ZprimeToZH_ZToTauTau_HToBB_M6000 \
 --bkg all --json CrossSectionZttHbb.json \
 --base /grid_mnt/data__data.polcms/cms/cuisset/cmt/ --ver bul_2016_ZttHbb_v12,bul_2016_HIPM_ZttHbb_v12,bul_2017_ZttHbb_v12,bul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_OC90_sr --prd prod_241213c_DNNtraining2 --stat_prd prod_241024 | tee ZttHbb.log
```

```bash
python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZZbbtt_0 --run 0 --num 0 | tee 2025_01_09/DNNWeight_ZZbbtt_0/training_0.log
python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZZbbtt_0 --run 0 --num 1 | tee 2025_01_09/DNNWeight_ZZbbtt_0/training_1.log

python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZbbHtt_0 --run 0 --num 0 | tee 2025_01_09/DNNWeight_ZbbHtt_0/training_0.log
python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZbbHtt_0 --run 0 --num 1 | tee 2025_01_09/DNNWeight_ZbbHtt_0/training_1.log

python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZttHbb_0 --run 0 --num 0 | tee 2025_01_09/DNNWeight_ZttHbb_0/training_0.log
python3 -u TrainDNN.py --out 2025_01_09/DNNWeight_ZttHbb_0 --run 0 --num 1 | tee 2025_01_09/DNNWeight_ZttHbb_0/training_1.log
```

```bash
python3 TestDNN.py --out 2025_01_09/DNNWeight_ZZbbtt_0 --run 0

python3 TestDNN.py --out 2025_01_09/DNNWeight_ZbbHtt_0 --run 0

python3 TestDNN.py --out 2025_01_09/DNNWeight_ZttHbb_0 --run 0
```

```bash
conda activate lumin_3.7
python3 SaveDNN.py --out 2025_01_09/DNNWeight_ZZbbtt_0 --run 0 --name ResZZbbtt --fold 2025-01-09

python3 SaveDNN.py --out 2025_01_09/DNNWeight_ZbbHtt_0 --run 0 --name ResZbbHtt --fold 2025-01-09

python3 SaveDNN.py --out 2025_01_09/DNNWeight_ZttHbb_0 --run 0 --name ResZttHbb --fold 2025-01-09

```

</details>

<details>
<summary>FullRun2 ResDNN commands</summary>

```bash
python3 ProduceDNNInputs.py --out 2024_04_29/DNNWeight_ZZbbtt_0 \
 --sig GluGluToXToZZTo2B2Tau_M200,GluGluToXToZZTo2B2Tau_M210,GluGluToXToZZTo2B2Tau_M220,GluGluToXToZZTo2B2Tau_M230,GluGluToXToZZTo2B2Tau_M240,GluGluToXToZZTo2B2Tau_M250,\
GluGluToXToZZTo2B2Tau_M260,GluGluToXToZZTo2B2Tau_M270,GluGluToXToZZTo2B2Tau_M280,GluGluToXToZZTo2B2Tau_M300,GluGluToXToZZTo2B2Tau_M320,GluGluToXToZZTo2B2Tau_M350,GluGluToXToZZTo2B2Tau_M360,\
GluGluToXToZZTo2B2Tau_M400,GluGluToXToZZTo2B2Tau_M450,GluGluToXToZZTo2B2Tau_M500,GluGluToXToZZTo2B2Tau_M550,GluGluToXToZZTo2B2Tau_M600,GluGluToXToZZTo2B2Tau_M650,GluGluToXToZZTo2B2Tau_M700,\
GluGluToXToZZTo2B2Tau_M750,GluGluToXToZZTo2B2Tau_M800,GluGluToXToZZTo2B2Tau_M850,GluGluToXToZZTo2B2Tau_M900,GluGluToXToZZTo2B2Tau_M1000,GluGluToXToZZTo2B2Tau_M1100,GluGluToXToZZTo2B2Tau_M1200,\
GluGluToXToZZTo2B2Tau_M1300,GluGluToXToZZTo2B2Tau_M1400,GluGluToXToZZTo2B2Tau_M1500,GluGluToXToZZTo2B2Tau_M1600,GluGluToXToZZTo2B2Tau_M1700,GluGluToXToZZTo2B2Tau_M1800,GluGluToXToZZTo2B2Tau_M1900,\
GluGluToXToZZTo2B2Tau_M2000,GluGluToXToZZTo2B2Tau_M2200,GluGluToXToZZTo2B2Tau_M2400,GluGluToXToZZTo2B2Tau_M2500,GluGluToXToZZTo2B2Tau_M2600,GluGluToXToZZTo2B2Tau_M2800,GluGluToXToZZTo2B2Tau_M3000,\
GluGluToXToZZTo2B2Tau_M3500,GluGluToXToZZTo2B2Tau_M4000,GluGluToXToZZTo2B2Tau_M4500,GluGluToXToZZTo2B2Tau_M5000 \
 --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2016_ZZ_v12,ul_2016_HIPM_ZZ_v12,ul_2017_ZZ_v12,ul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_240318 --stat_prd prod_240305 --eos True

python3 ProduceDNNInputs.py --out 2024_04_29/DNNWeight_ZbbHtt_0 \
 --sig ZprimeToZH_ZToBB_HToTauTau_M600,ZprimeToZH_ZToBB_HToTauTau_M800,ZprimeToZH_ZToBB_HToTauTau_M1000,ZprimeToZH_ZToBB_HToTauTau_M1200,ZprimeToZH_ZToBB_HToTauTau_M1400,\
ZprimeToZH_ZToBB_HToTauTau_M1600,ZprimeToZH_ZToBB_HToTauTau_M1800,ZprimeToZH_ZToBB_HToTauTau_M2000,ZprimeToZH_ZToBB_HToTauTau_M2500,ZprimeToZH_ZToBB_HToTauTau_M3000,\
ZprimeToZH_ZToBB_HToTauTau_M3500,ZprimeToZH_ZToBB_HToTauTau_M4000 \
 --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZbbHtt_v12,ul_2016_HIPM_ZbbHtt_v12,ul_2017_ZbbHtt_v12,ul_2018_ZbbHtt_v12 \
 --cat cat_ZbbHtt_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True

python3 ProduceDNNInputs.py --out 2024_04_29/DNNWeight_ZttHbb_0 \
 --sig ZprimeToZH_ZToTauTau_HToBB_M600,ZprimeToZH_ZToTauTau_HToBB_M800,ZprimeToZH_ZToTauTau_HToBB_M1000,ZprimeToZH_ZToTauTau_HToBB_M1200,ZprimeToZH_ZToTauTau_HToBB_M1400,\
ZprimeToZH_ZToTauTau_HToBB_M1600,ZprimeToZH_ZToTauTau_HToBB_M1800,ZprimeToZH_ZToTauTau_HToBB_M2000,ZprimeToZH_ZToTauTau_HToBB_M2500,ZprimeToZH_ZToTauTau_HToBB_M3000,\
ZprimeToZH_ZToTauTau_HToBB_M3500,ZprimeToZH_ZToTauTau_HToBB_M4000 \
 --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2016_ZttHbb_v12,ul_2016_HIPM_ZttHbb_v12,ul_2017_ZttHbb_v12,ul_2018_ZttHbb_v12 \
 --cat cat_ZttHbb_elliptical_cut_90_sr --prd prod_240312_DNNinput --stat_prd prod_240305 --eos True
```

```bash
python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZZbbtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZZbbtt_0 --run 0 --num 1

python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZbbHtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZbbHtt_0 --run 0 --num 1

python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZttHbb_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_04_29/DNNWeight_ZttHbb_0 --run 0 --num 1
```

```bash
python3 TestDNN.py --out 2024_04_29/DNNWeight_ZZbbtt_0 --run 0
python3 TestDNN.py --out 2024_04_29/DNNWeight_ZbbHtt_0 --run 0 
python3 TestDNN.py --out 2024_04_29/DNNWeight_ZttHbb_0 --run 0 
```

```bash
conda activate lumin_3.7
python3 SaveDNN.py --out 2024_04_29/DNNWeight_ZZbbtt_0 --run 0 --name ResZZbbtt --fold 2024-04-29
python3 SaveDNN.py --out 2024_04_29/DNNWeight_ZbbHtt_0 --run 0 --name ResZbbHtt --fold 2024-04-29
python3 SaveDNN.py --out 2024_04_29/DNNWeight_ZttHbb_0 --run 0 --name ResZttHbb --fold 2024-04-29
```

</details>

<details>
<summary>2018 commands</summary>

```bash
python3 ProduceDNNInputs.py --out 2024_03_26/DNNWeight_ZZbbtt_0 \
 --sig ggXZZbbtt_M200,ggXZZbbtt_M300,ggXZZbbtt_M400,ggXZZbbtt_M500,ggXZZbbtt_M600,ggXZZbbtt_M700,ggXZZbbtt_M800,ggXZZbbtt_M900,\
ggXZZbbtt_M1000,ggXZZbbtt_M1100,ggXZZbbtt_M1200,ggXZZbbtt_M1300,ggXZZbbtt_M1400,ggXZZbbtt_M1500,ggXZZbbtt_M2000,ggXZZbbtt_M3000 \
 --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2018_ZZ_v12 \
 --cat cat_ZZ_elliptical_cut_90_sr --prd prod_240318 --stat_prd prod_240305 --eos True
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
```

```bash
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_0 --run 0 --num 1

python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_0 --run 0 --num 1

python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZttHbb_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_03_26/DNNWeight_ZttHbb_0 --run 0 --num 1
```

```bash
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_0 --run 0
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_0 --run 0 
python3 TestDNN.py --out 2024_03_26/DNNWeight_ZttHbb_0 --run 0 
```

```bash
conda activate lumin_3.7
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZZbbtt_0 --run 0 --name ResZZbbtt --fold 2024-03-26
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZbbHtt_0 --run 0 --name ResZbbHtt --fold 2024-03-26
python3 SaveDNN.py --out 2024_03_26/DNNWeight_ZttHbb_0 --run 0 --name ResZttHbb --fold 2024-03-26
```

</details>

## Inputs

The inputs are cretaed via the new Framework developed by Jaime: https://github.com/elenavernazza/hhbbtt-analysis
The information about the cross section is read in the config file and saved into the `CrossSection.json` file.

```bash
python3 GetSamplesConfig.py
```

The ntuples are converted to hdf5 files for the DNN training.

```bash
(seteos)

python3 ProduceDNNInputs.py --out 2024_02_15/DNNWeight_ZZbbtt_0 --sig zz_sl_signal --bkg all --json CrossSectionZZ.json \
 --base /data_CMS/cms/vernazza/cmt/ --ver ul_2018_ZZ_v10 \
 --cat cat_ZZ_elliptical_cut_80_sr --prd prod_240207 --stat_prd prod_240128 --eos True
python3 ProduceDNNInputs.py --out 2024_02_15/DNNWeight_ZbbHtt_0 --sig zh_zbb_htt_signal --bkg all --json CrossSectionZbbHtt.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZbbHtt_v10 \
 --cat cat_ZbbHtt_elliptical_cut_90_sr --prd prod_240128 --stat_prd prod_240128 --eos True
python3 ProduceDNNInputs.py --out 2024_02_15/DNNWeight_ZttHbb_0 --sig zh_ztt_hbb_signal --bkg all --json CrossSectionZttHbb.json \
 --base /data_CMS/cms/cuisset/cmt/ --ver ul_2018_ZttHbb_v10 \
 --cat cat_ZttHbb_elliptical_cut_90_sr --prd prod_240128 --stat_prd prod_240128 --eos True
```

Inside the script, the events are weighted by the cross section, the generator level weights and the detector related weights. The negative weights are removed.
The event weight is then renormalized in order to give the same importance to the signal sample and to the sum of backgrounds.

## Training

The training needs to be performed on machines with GPU. At LLR, use the llrai machine.
Two trainings have to performed for the even and odd event numbers: they can be run at the same time, selecting the first GPU device for the first training and the second GPU device for the second training (already autmatically selected in the script).

```bash
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZZbbtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZZbbtt_0 --run 0 --num 1
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZbbHtt_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZbbHtt_0 --run 0 --num 1
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZttHbb_0 --run 0 --num 0
python3 TrainDNN.py --out 2024_02_15/DNNWeight_ZttHbb_0 --run 0 --num 1
```

## Testing

```bash
python3 TestDNN.py --out 2024_02_15/DNNWeight_ZZbbtt_0 --run 0 
python3 TestDNN.py --out 2024_02_15/DNNWeight_ZbbHtt_0 --run 0
python3 TestDNN.py --out 2024_02_15/DNNWeight_ZttHbb_0 --run 0
```

## Export

The export is a delicate process and requires a particular environment. If these instructions are not followed, the exportation still works BUT the output has a format that cannot be used into the default analysis framework (.pb are not files, but folders).
To create the environment:

```bash
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

```bash
python3 SaveDNN.py --out 2024_02_15/DNNWeight_ZZbbtt_0 --run 0 --name ZZbbtt
python3 SaveDNN.py --out 2024_02_15/DNNWeight_ZbbHtt_0 --run 0 --name ZbbHtt
python3 SaveDNN.py --out 2024_02_15/DNNWeight_ZttHbb_0 --run 0 --name ZttHbb
```