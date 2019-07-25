from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# import _init_paths
from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.loss import JointsMSELoss
from lib.core.function import validate
from lib.utils.utils import create_logger
from collections import OrderedDict

import lib.dataset as dataset
import lib.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_neuron_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        checkpoint = torch.load(config.TEST.MODEL_FILE)
        if isinstance(checkpoint, OrderedDict):
            state_dict_old = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_old = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(config.TEST.MODEL_FILE))

        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith('module.'):
                # state_dict[key[7:]] = state_dict[key]
                # state_dict.pop(key)
                state_dict[key[7:]] = state_dict_old[key]
            else:
                state_dict[key] = state_dict_old[key]

        model.load_state_dict(state_dict)
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'valid_global_steps': 0,
        'vis_global_steps': 0,
    }

    dump_input = torch.rand((config.TEST.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion)
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    n_sources = [1, 2, 4, 8, 16, 32, 64]
    for ns in n_sources:
        # Data loading code
        normalize = transforms.Normalize(mean=[0.1440, 0.1440, 0.1440],
                                         std=[19.0070, 19.0070, 19.0070])

        valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TEST_SET,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            n_sources=ns
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST.BATCH_SIZE*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
