import json

def get_xs_from_txt(dataset_name, file_path):
    with open(file_path, 'r') as file:
        found_sample = False
        found_xs = False
        for line in file.readlines():
            if found_xs: break
            if not found_sample and f'Dataset("{dataset_name}"' in line:
                found_sample = True
            if found_sample and 'xs=' in line:
                found_xs = True
                xs = float(line.split('xs=')[1].split(',')[0])
    return xs

def get_xs_dict(sample_list, file_path, json_path):
    
    xs_dict = {}

    for sample in sample_list:
        xs_dict[sample] = get_xs_from_txt(sample, file_path)

    with open(json_path, 'w') as json_file:
        json.dump(xs_dict, json_file, indent=2)

    return xs_dict

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--cfg",       dest="cfg",      default='/grid_mnt/data__data.polcms/cms/vernazza/FrameworkNanoAOD/hhbbtt-analysis/config/ul_2018_ZZ_v10.py')
    parser.add_option("--samples",   dest="samples",  default='ggXZZbbtt_M300,ggXZZbbtt_M600,ggXZZbbtt_M700,zz_sl_signal,dy,tt_dl,tt_sl,tt_fh,wjets,ttw_lnu,ttw_qq,ttww,ttwz,ttwh,ttzh,ttz_llnunu,ttz_qq,ttzz,tth_bb,tth_nonbb,tth_tautau,zh_hbb_zll,wplush_htt,wminush_htt,ggH_ZZ,ggf_sm,zz_dl,zz_sl_background,zz_lnu,zz_qnu,wz_lllnu,wz_lnuqq,wz_llqq,wz_lnununu,ww_llnunu,ww_lnuqq,ww_qqqq,zzz,wzz,www,wwz,ewk_z,ewk_wplus,ewk_wminus,st_tw_antitop,st_tw_top,st_antitop,st_top')
    parser.add_option("--json",      dest="json",     default='CrossSection.json')
    (options, args) = parser.parse_args()

    file_path = options.cfg
    sample_list = options.samples.split(',')
    json_path = options.json
    print("\n ### INFO: Condifg file", file_path)
    print("\n ### INFO: Samples list", sample_list)
    print("\n ### INFO: Json file", json_path)
    print(" \n")

    get_xs_dict(sample_list, file_path, json_path)
