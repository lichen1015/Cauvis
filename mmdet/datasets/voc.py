# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

@DATASETS.register_module()
class DIORDataset(XMLDataset):
    """DIOR dataset for detection.

    Args:
        ann_subdir (str): Subdir where annotations are.
            Defaults to 'Annotations/Oriented Bounding Boxes/'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ann_type (str): Choose obb or hbb as ground truth.
            Defaults to `obb`.
    """

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
         'chimney', 'expressway-service-area', 'expressway-toll-station',
         'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
         'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
         'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metainfo['dataset_type'] = "DIOR"

@DATASETS.register_module()
class SdgodDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes': ('bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck'),
        # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        #  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metainfo['dataset_type'] = "SDGOD"