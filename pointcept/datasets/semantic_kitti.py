"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import random

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/semantic_kitti",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        sequence_length=5,
        concatenate_scans=True,
        stack_scans=False
    ):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.sequence_length = sequence_length
        self.concatenate_scans = concatenate_scans
        self.stack_scans = stack_scans
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        data_list = []
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]

        return data_list  # this contains a list of the binary file paths

    def get_time_list(self):

        time_list = []
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        for folder in seq_list:
            folder = str(folder).zfill(2)
            time_folder = os.path.join(self.data_root, "dataset", "sequences", folder)
            time_list += [
                os.path.join(time_folder, "times.txt")
            ]
        return time_list

    def map_data_to_sequence_index(self, data_index):
        if 0 <= data_index < 4541:  # index from sequence 00 chosen
            sequence = 0
            index = data_index
        elif 4541 <= data_index < 5642:
            sequence = 1
            index = data_index - 4541
        elif 5642 <= data_index < 10303:
            sequence = 2
            index = data_index - 5642
        elif 10303 <= data_index < 11104:
            sequence = 3
            index = data_index - 10303
        elif 11104 <= data_index < 11375:
            sequence = 4
            index = data_index - 11104
        elif 11375 <= data_index < 14136:
            sequence = 5
            index = data_index - 11375
        elif 14136 <= data_index < 15237:
            sequence = 6
            index = data_index - 14136
        elif 15237 <= data_index < 16338:
            sequence = 7
            index = data_index - 15237
        elif 16338 <= data_index < 17929:
            sequence = 9
            index = data_index - 16338
        elif 17929 <= data_index < 19130:
            sequence = 10
            index = data_index - 17929
        else:
            raise NotImplementedError

        return sequence, index

    def get_data(self, idx):

        data_index = idx % len(self.data_list)
        sequence, index = self.map_data_to_sequence_index(data_index)
        sequence = str(sequence).zfill(2)

        if index < self.sequence_length:
            index = random.randrange(self.sequence_length, self.sequence_length*5, 1)

        # Create list of scan indices
        scans = np.arange(index, index-self.sequence_length, -1)

        # Set up the appropriate paths/files
        scan_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "velodyne")
        time_file = open(os.path.join(self.data_root, "dataset", "sequences", sequence, "times.txt"))
        label_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "labels")

        # Create output arrays
        coordinates = np.empty([0, 3])
        remissions = np.empty([0, 1])
        time_data = (np.loadtxt(time_file, dtype=np.float32).reshape(-1, 1))
        times = np.empty([0, 1])
        labels = []

        # Load times for sequence
        selected_time = time_data[index]

        if self.concatenate_scans:
            for i, scan in enumerate(scans):
                # Open the scan file
                scan_filename = str(scan).zfill(6) + ".bin"
                scan_file = os.path.join(scan_path, scan_filename)
                with open(scan_file, "rb") as scan_data:
                    scan = np.fromfile(scan_data, dtype=np.float32).reshape(-1, 4)

                # Append new scan to coordinates
                coordinates = np.vstack((coordinates,
                                         scan[:, :3]
                                         )
                                        )
                # Append new remissions
                new_remissions = scan[:, -1].reshape([-1, 1])
                remissions = np.vstack((remissions,
                                        new_remissions
                                        )
                                       )
                # Append times for the scan
                times = np.vstack((times,
                                   np.full_like(new_remissions,
                                                (time_data[scans[i]] - selected_time)
                                                )
                                   )
                                  )

                # Open the labels file
                label_file = scan_file.replace("velodyne", "labels").replace(".bin", ".label")
                if os.path.exists(label_file):
                    with open(label_file, "rb") as a:
                        segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                        segment = np.vectorize(self.learning_map.__getitem__)(
                            segment & 0xFFFF
                        ).astype(np.int32)
                else:
                    segment = np.zeros(scan.shape[0]).astype(np.int32)

                # Append the labels for the scan
                labels = np.hstack((labels,
                                    segment
                                    )
                                   ).astype(np.int32)

        elif self.stack_scans:
            # do the Semantic Kitti API "generate sequential" transformation
            raise NotImplementedError
        else:
            # do the original data loader from pointcept
            raise NotImplementedError
        # print("\nCoordinates shape: " + str(np.shape(coordinates)))
        # print("\nRemissions shape: " + str(np.shape(remissions)))
        # print("\nTimes shape: " + str(np.shape(times)))
        # print("\nLabels shape: " + str(np.shape(labels)))
        data_dict = dict(coord=coordinates, strength=remissions, time=times, segment=labels)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
