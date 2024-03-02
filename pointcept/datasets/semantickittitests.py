

import semantic_kitti


dataset = semantic_kitti.SemanticKITTIDataset(
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
)

