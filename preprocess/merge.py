import os

import numpy as np
import shutil

#vctk = [225, 228, 229, 230, 231, 233, 236, 239, 240, 244, 226, 227, 232, 243, 254, 256, 258, 259, 262, 272]
vctk = [sp.replace('p', '') for sp in os.listdir('/project/sughosh/dataset/VCTK-24k/') if os.path.isdir(os.path.join('/project/sughosh/dataset/VCTK-24k/', sp))]
voxceleb = ['10935', #M
                     '11184',#M
                     '10977',#M
                     '10945',#M
                     '10520',#M
                     '10539',#M
                     '10921',#M
                     '10326',#M
                     '10571',#F
                     '10393',#M
                     '10443',#M
                     '11173',#M
                     '10870',#F
                     '10826',#F
                     '10907',#F
                     '10167',#F
                     '10856',#F
                     '11003',#F
                     '10938',#F
                     '10696',#F
                     #'10696',#F
                     #'10616'#F
                     #'10317'#F,
                     #'10997',
                     #'10352',
                     #'10715'
                      ]
__SPEAKERS__ = vctk + voxceleb

def generate_train_test_split(trn_flnm='train_list.txt', val_flnm='val_list.txt'):
    for spk in __SPEAKERS__:
        data_list = []
        files = os.listdir(os.path.join(__OUTPATH__, 'p'+str(spk)))
        for file in files:
            if file.endswith(".wav"):
                    data_list.append(os.path.join(__OUTPATH__, 'p'+str(spk), file))

        len_data_list = len(data_list)
        test_idx = int(0.1 * len_data_list)
        train_data = data_list[test_idx:]
        test_data = data_list[0:test_idx]

        # write to file

        file_str = ""
        for item in train_data:
            file_str += item + "|" + str(int(__SPEAKERS__.index(spk))) + '\n'
        text_file = open(__OUTPATH__ + "/"+trn_flnm, "a")
        text_file.write(file_str)
        text_file.close()

        file_str = ""
        for item in test_data:
            file_str += item + "|" + str(int(__SPEAKERS__.index(spk))) + '\n'
        text_file = open(__OUTPATH__ + "/"+val_flnm, "a")
        text_file.write(file_str)
        text_file.close()

if __name__ == '__main__':
    vctk_dir = '/project/sughosh/dataset/VCTK-24k'
    vox_dir = '/project/sughosh/dataset/voxceleb-24k'
    __OUTPATH__ = '/project/sughosh/dataset/vctkall-voxceleb-24k'
    #print(__SPEAKERS__)
    sp = [int(sp) for sp in __SPEAKERS__]
    print(sp)
    # print(len(sp))
    #
    # for s in vctk:
    #      shutil.copytree(os.path.join(vctk_dir, 'p'+str(s)), os.path.join(__OUTPATH__, 'p'+str(s)), dirs_exist_ok=True)
    #
    # for s in voxceleb:
    #      shutil.copytree(os.path.join(vox_dir, 'p'+str(s)), os.path.join(__OUTPATH__, 'p'+str(s)))
    #


    #generate_train_test_split('train_list_allvctk_vox20.txt', 'val_list_allvctk_vox20.txt')



