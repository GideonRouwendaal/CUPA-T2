import numpy as np
import os
import subprocess
import glob


def train_val_test_split(files, nr_val_datasets, nr_train_datasets):
    """
    Custom dataset split based on exclusion criteria.

    Parameters
    ----------
    files : list of str
        List of file names.
    nr_val_datasets : int
        Number of datasets for validation.
    nr_train_datasets : int
        Number of datasets for training.

    Returns
    -------
    train_files, val_files, test_files : list of str
        Lists containing file names for training, validation, and testing,
        respectively.

    Notes
    -----
    - Sorts the subjects "SQ-struct-30", "SQ-struct-31", "SQ-struct-32"
     and "SQ-struct-42" into the train set, since they are acquired without
     motion data.
    """

    train_files = [f for f in files if
                   any(s in f for s in ["SQ-struct-30", "SQ-struct-31",
                                        "SQ-struct-32", "SQ-struct-42"])]
    files = [f for f in files if f not in train_files]
    val_files = files[:nr_val_datasets]
    ind_test = nr_val_datasets + nr_train_datasets - len(train_files)
    test_files = files[ind_test:]
    train_files.extend(f for f in files[nr_val_datasets:ind_test])

    return train_files, val_files, test_files


def create_dir(folder):
    """Create a directory if it does not exist."""

    if not os.path.exists(folder):
        os.makedirs(folder)
    return 0


class SymbolicLinks:
    def __init__(self):
        super(SymbolicLinks, self).__init__()


    def create_link(self, file_name, link_name):
        """Create a symbolic link from a source file to a destination link."""

        subprocess.run('ln -s ' + file_name + ' ' + link_name, shell=True)
        return 0


    # def loop_through_subjects(self, folder_out, machine, folder_in):
    #     """Loops through subjects and creates symbolic links"""
    #     for set in ['val', 'train', 'test', 'test-patterns', 'test-tfe', 'test-extreme']:
    #         if os.path.exists("{}recon_{}_{}".format(folder_out, set,
    #                                                  machine)):
    #             print("ERROR: Directory {}recon_{}_{} already "
    #                   "exists.".format(folder_out, set, machine))
    #             continue

    #         subjects = np.loadtxt("{}files_recon_{}.txt".format(folder_out,
    #                                                             set),
    #                               dtype=str)
    #         if subjects.size == 1:
    #             subjects = [subjects]

    #         for subject in subjects:
    #             create_dir("{}recon_{}/raw_data/".format(folder_out,
    #                                                         set))
    #             create_dir("{}recon_{}/brain_masks/".format(folder_out,
    #                                                            set))

    #             # raw data:
    #             file = glob.glob(
    #                 "{}{}/**wip_t2s**_sg_fV4.mat".format(folder_in, subject)
    #             )[0]
    #             link_name = "{}recon_{}/raw_data/{}_{}".format(
    #                 folder_out, set,
    #                 os.path.basename(os.path.dirname(file)),
    #                 os.path.basename(file)
    #             )
    #             self.create_link(file, link_name)

    #             # brain mask:
    #             file_bm = ("{}/output/sub-{}/task-still/qBOLD/T1w_coreg/"
    #                        "rcBrMsk_CSF.nii".format(
    #                 folder_in.replace("converted_raw", "bids_from_raw"),
    #                 subject)
    #             )
    #             link_name_bm = (link_name.replace(
    #                 "/raw_data/", "/brain_masks/").replace(
    #                 ".mat", "_bm.nii"))
    #             self.create_link(file_bm, link_name_bm)

    #     for set_motion, set in zip(
    #             ['val_motion', 'train_motion', 'test_motion',
    #              'test-patterns_motion', 'test-tfe_motion', 'test-extreme_motion'],
    #             ['val', 'train', 'test', 'test-patterns', 'test-tfe', 'test-extreme']
    #     ):
    #         if os.path.exists("{}recon_{}".format(folder_out,
    #                                                  set_motion)):
    #             print("ERROR: Directory {}recon_{} already exists.".format(
    #                 folder_out, set_motion)
    #             )
    #             continue

    #         subjects = np.loadtxt("{}files_recon_{}.txt".format(
    #             folder_out, set), dtype=str
    #         )
    #         if subjects.size == 1:
    #             subjects = [subjects]

    #         for subject in subjects:
    #             if subject not in ["SQ-struct-30", "SQ-struct-31",
    #                                "SQ-struct-32", "SQ-struct-42"]:
    #                 create_dir("{}recon_{}/raw_data/".format(
    #                     folder_out, set_motion)
    #                 )
    #                 create_dir("{}recon_{}/brain_masks/".format(
    #                     folder_out, set_motion)
    #                 )

    #                 # raw data:
    #                 if "tfe" not in set_motion:
    #                     file = glob.glob(
    #                         "{}{}/**wip_t2s**_sg_f_moveV4.mat".format(
    #                             folder_in, subject)
    #                     )[0]
    #                 else:
    #                     file = glob.glob(
    #                         "{}{}/**wip_t2s**_sg_tfe_f_moveV4.mat".format(
    #                             folder_in, subject)
    #                     )[0]
    #                 link_name = "{}recon_{}/raw_data/{}_{}".format(
    #                     folder_out, set_motion,
    #                     os.path.basename(os.path.dirname(file)),
    #                     os.path.basename(file)
    #                 )
    #                 self.create_link(file, link_name)

    #                 # brain mask:
    #                 file_bm = ("{}/output/sub-{}/task-move/qBOLD/T1w_coreg/"
    #                            "rcBrMsk_CSF.nii".format(
    #                     folder_in.replace("converted_raw", "bids_from_raw"),
    #                     subject)
    #                 )
    #                 link_name_bm = (link_name.replace(
    #                     "/raw_data/", "/brain_masks/").replace(
    #                     ".mat", "_bm.nii")
    #                 )
    #                 self.create_link(file_bm, link_name_bm)


    #                 # also the corresponding GT:
    #                 file = glob.glob(
    #                     "{}{}/**wip_t2s**_sg_fV4.mat".format(
    #                         folder_in, subject)
    #                 )[0]
    #                 link_name = "{}recon_{}/raw_data/{}_{}".format(
    #                     folder_out, set_motion,
    #                     os.path.basename(os.path.dirname(file)),
    #                     os.path.basename(file)
    #                 )
    #                 self.create_link(file, link_name)

    #                 # brain mask:
    #                 file_bm = ("{}/output/sub-{}/task-still/qBOLD/T1w_coreg/"
    #                            "rcBrMsk_CSF.nii".format(
    #                     folder_in.replace("converted_raw", "bids_from_raw"),
    #                     subject)
    #                 )
    #                 link_name_bm = link_name.replace(
    #                     "/raw_data/", "/brain_masks/").replace(
    #                     ".mat", "_bm.nii"
    #                 )
    #                 self.create_link(file_bm, link_name_bm)


    def loop_through_subjects(self, folder_out, machine, folder_in):
        """
        Loops through subjects and creates symbolic links for reconstruction (no motion).
        - Assumes files are organized as: folder_in/set/subject/t2s_gre_fr.hf and seg_brain_reg-to-t2s.nii
        """
        for set in ['val', 'train', 'test', 'test-patterns', 'test-tfe', 'test-extreme']:
            out_dir = f"{folder_out}recon_{set}_{machine}"
            if os.path.exists(out_dir):
                print(f"ERROR: Directory {out_dir} already exists.")
                continue

            subjects = np.loadtxt(f"{folder_out}files_recon_{set}.txt", dtype=str)
            if subjects.size == 1:
                subjects = [subjects]

            for subject in subjects:
                raw_data_dir = f"{folder_out}recon_{set}/raw_data/"
                brain_mask_dir = f"{folder_out}recon_{set}/brain_masks/"
                create_dir(raw_data_dir)
                create_dir(brain_mask_dir)

                # Find raw data file (*.t2s_gre_fr.hf)
                files = glob.glob(os.path.join(folder_in, set, subject, "*t2s_gre_fr.hf"))
                if files:
                    file = files[0]
                    link_name = os.path.join(raw_data_dir, f"{subject}_{os.path.basename(file)}")
                    self.create_link(file, link_name)
                else:
                    print(f"No raw data file found for subject {subject} in set {set}")

                # Find brain mask file (seg_brain_reg-to-t2s.nii)
                file_bm = os.path.join(folder_in, set, subject, "seg_brain_reg-to-t2s.nii")
                if os.path.exists(file_bm):
                    link_name_bm = os.path.join(brain_mask_dir, f"{subject}_seg_brain_reg-to-t2s.nii")
                    self.create_link(file_bm, link_name_bm)
                else:
                    print(f"No brain mask file found for subject {subject} in set {set}")


    def loop_through_subjects_simulated(self, folder_out, machine, folder_in,
                                        use_sim_key=False):

        sim_key = ("-"+os.path.basename(os.path.normpath(folder_in))
                   if use_sim_key else "")

        for set in ['val', 'train', 'test']:
            set_motion = "{}-simulated{}_motion".format(set, sim_key)

            if os.path.exists("{}recon_{}".format(folder_out,
                                                     set_motion)):
                print("ERROR: Directory {}recon_{} already exists.".format(
                    folder_out, set_motion)
                )
                continue

            subjects_ = np.loadtxt("{}files_recon_{}.txt".format(
                folder_out, set), dtype=str
            )
            if subjects_.size == 1:
                subjects_ = [subjects_]

            subjects = []
            for subject in subjects_:
                for s in [s for s in os.listdir(folder_in) if subject in s]:
                    subjects.append(s)

            if len(subjects) == 0:
                print("ERROR: No simulated subjects found for set {}.".format(set))
                continue

            create_dir("{}recon_{}/raw_data/".format(
                folder_out, set_motion)
            )
            create_dir("{}recon_{}/brain_masks/".format(
                folder_out, set_motion)
            )

            for subject in subjects:
                # raw data:
                file = glob.glob(
                    "{}{}/**wip_t2s**_sg_f_moveV4.h5".format(
                        folder_in, subject)
                )[0]
                link_name = "{}recon_{}/raw_data/{}_{}".format(
                    folder_out, set_motion,
                    os.path.basename(os.path.dirname(file)),
                    os.path.basename(file)
                )
                self.create_link(file, link_name)

                # brain mask:
                file_bm = (
                    "{}/output/sub-{}/task-still/qBOLD/T1w_coreg/rcBrMsk_CSF"
                    ".nii".format(
                        os.path.dirname(
                            os.path.normpath(folder_in)
                        ).replace("simulated_raw", "bids_from_raw"),
                        subject.split('-sim')[0])
                )
                link_name_bm = (link_name.replace(
                    "/raw_data/", "/brain_masks/").replace(
                    ".h5", "_bm.nii")
                )
                self.create_link(file_bm, link_name_bm)


                # also the corresponding GT:
                file = glob.glob(
                    "{}{}/**wip_t2s**_sg_fV4.mat".format(
                        folder_in, subject)
                )[0]
                link_name = "{}recon_{}/raw_data/{}_{}".format(
                    folder_out, set_motion,
                    os.path.basename(os.path.dirname(file)),
                    os.path.basename(file)
                )
                self.create_link(file, link_name)

                # brain mask:
                file_bm = (
                    "{}/output/sub-{}/task-still/qBOLD/T1w_coreg/rcBrMsk_CSF"
                    ".nii".format(
                        os.path.dirname(
                            os.path.normpath(folder_in)
                        ).replace("simulated_raw", "bids_from_raw"),
                        subject.split('-sim')[0])
                )
                link_name_bm = link_name.replace(
                    "/raw_data/", "/brain_masks/").replace(
                    ".mat", "_bm.nii"
                )
                self.create_link(file_bm, link_name_bm)
