# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Optional, Sequence, Union

import numpy as np
from mmengine.config import Config, ConfigDict
from mmengine.dataset import Compose
from mmengine.model import BaseModel

from mmseg.datasets.transforms import LoadAnnotations

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseModel,
                   test_pipeline: Optional[Compose]):

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            test_pipeline[0]['type'] = 'mmseg.LoadImageFromNDArray'

    for t in test_pipeline:
        if t['type'] in ['LoadAnnotations', LoadAnnotations]:
            test_pipeline.remove(t)

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch


def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)
