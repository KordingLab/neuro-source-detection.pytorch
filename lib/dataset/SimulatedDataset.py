import logging
import os
import json
import numpy as np
# from scipy.io import loadmat, savemat
import copy

from torch.utils.data import Dataset

from lib.utils.utils import get_source, evalpotential, generate_heatmaps

logger = logging.getLogger(__name__)

class SimulatedDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform):
        self.cfg = cfg
        self.is_train = is_train

        self.root = root
        self.image_set = image_set

        self.is_train = is_train

        self.transform = transform

        self.patch_width = cfg.MODEL.IMAGE_SIZE[0]
        self.patch_height = cfg.MODEL.IMAGE_SIZE[1]

        self.db = self._get_db()
        self.db_length = len(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def __getitem__(self, idx):
        the_db = copy.deepcopy(self.db[idx])

        image = evalpotential(the_db['mesh'], the_db['sources'])

        image = image.reshape((self.patch_height, self.patch_width)) + \
            self.cfg.MODEL.VAR_NOISE*np.random.randn(self.patch_height, self.patch_width)

        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1).astype(np.float32)

        image = self.transform(image)

        unnormalized_sources = the_db['sources'].transpose()[:, 1:3].copy()
        unnormalized_sources[:, 0] *= self.cfg.MODEL.OUTPUT_SIZE[0]
        unnormalized_sources[:, 1] *= self.cfg.MODEL.OUTPUT_SIZE[1]

        heatmap_target = generate_heatmaps(
            unnormalized_sources,
            self.cfg.MODEL.OUTPUT_SIZE[0],
            self.cfg.MODEL.OUTPUT_SIZE[1]
        )

        meta = {
            'sources': unnormalized_sources
        }

        return image, heatmap_target, meta

    def _get_db(self):
        gt_db = []
        self.db_length = int(self.cfg.TRAIN.NUM_SAMPLES) if self.is_train else int(self.cfg.TEST.NUM_SAMPLES)
        n_sources = 30
        for i in range(self.db_length):
            mesh, sources = get_source(self.patch_width, self.patch_height,
                self.cfg.MODEL.DEPTH, n_sources, self.cfg.MODEL.VAR_NOISE)

            # image.view(-1).repeat(3).view(self.patch_width, self.patch_height, 3)
            # image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)

            # image = self.transform(image)

            # sources = sources.transpose()

            # heatmap_target = generate_heatmaps(sources, self.patch_height, self.patch_width)

            gt_db.append({
                'mesh': mesh,
                'sources': sources
            })

        return gt_db

    def __len__(self):
        return self.db_length
