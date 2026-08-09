"""
Script to create symbolic links to data sources (to enable easier handling
when training on different machines)
"""
import os.path
import glob
import numpy as np
from utils import *

# input and output folders:
folder_in_WS = ("/home/iml/gideon.rouwendaal/PHIMO2/data/mr_data/data_folder")
folder_in_WS_sim = ("/PATH1/TO/SIMULATED/DATA/")
folder_out_WS = "/home/iml/gideon.rouwendaal/PHIMO2/data/mr_data/links_to_data/"
folder_in_cluster = ("/PATH2/TO/CONVERTED/RAW/DATA/")
folder_in_cluster_sim = ("/PATH1/TO/SIMULATED/DATA/")
folder_out_cluster = ("/./data/links_to_data/")
nr_val_datasets = 3
nr_train_datasets = 6

# what to run:
run_WS = True
new_train_test_split = True
run_cluster = False
simulated_data = False


SL = SymbolicLinks()

if run_WS:
    if new_train_test_split:
        files = glob.glob(folder_in_WS+"**/**wip_t2s**_sg_fV4.mat")
        files = glob.glob(folder_in_WS + "/**/*t2s_gre_fr.hf", recursive=True)

        # exclude subject 36 due to problem with SENSE Ref scan:
        files = [f for f in files if "SQ-struct-36" not in f]

        np.random.shuffle(files)

        (train_files, val_files,
         test_files) = train_val_test_split(files, nr_val_datasets,
                                            nr_train_datasets)

        train_files = [f for f in files if "train" in f]
        val_files = [f for f in files if "val" in f]
        test_files = [f for f in files if "test" in f]

        val_subjects, train_subjects, test_subjects = [], [], []
        for files, set, subjects in zip([val_files, train_files, test_files],
                                        ['val', 'train', 'test'],
                                        [val_subjects, train_subjects,
                                         test_subjects]):
            if os.path.exists("{}files_recon_{}.txt".format(folder_out_WS,
                                                            set)):
                print(
                    "ERROR: {}files_recon_{}.txt already exists.".format(
                        folder_out_WS, set)
                )
                continue
            for file in files:
                subjects.append(os.path.basename(os.path.dirname(file)))

            np.savetxt("{}files_recon_{}.txt".format(folder_out_WS, set),
                       subjects, fmt="%s")

    if not simulated_data:
        SL.loop_through_subjects(folder_out_WS, "WS", folder_in_WS)
    else:
        SL.loop_through_subjects_simulated(folder_out_WS, "WS",
                                           folder_in_WS_sim)


if run_cluster:
    if not simulated_data:
        SL.loop_through_subjects(folder_out_cluster, "cluster",
                                 folder_in_cluster)
    else:
        SL.loop_through_subjects_simulated(folder_out_cluster, "cluster",
                                           folder_in_cluster_sim)

print('Done creating symbolic links')
