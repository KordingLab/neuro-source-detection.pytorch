# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch
import torch.autograd as autograd

from lib.core.config import get_model_name
# from lib.core.evaluate import accuracy
# from core.inference import get_final_preds, get_final_integral_preds
# from utils.transforms import flip_back
# from utils.vis import save_debug_images
# from utils.vis_plain_keypoint import vis_mpii_keypoints
# from utils.integral import softmax_integral_tensor


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)

        target = target.cuda(non_blocking=True)
        loss = criterion(output, target, None)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        # acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']

            writer.add_scalar('train_loss', losses.val, global_steps)

            # input_image = input.detach().cpu().numpy()[0]
            # min_val = input_image.min()
            # max_val = input_image.max()
            # input_image = (input_image - min_val) / (max_val - min_val)
            # heatmap_target = target.detach().cpu().numpy()[0]
            # heatmap_pred = output.detach().cpu().numpy()[0]
            # heatmap_pred[heatmap_pred < 0.0] = 0
            # heatmap_pred[heatmap_pred > 1.0] = 1.0

            # writer.add_image('input_recording', input_image, global_steps,
            #     dataformats='CHW')
            # writer.add_image('heatmap_target', heatmap_target, global_steps,
            #     dataformats='CHW')
            # writer.add_image('heatmap_pred', heatmap_pred, global_steps,
            #     dataformats='CHW')

            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, meta) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)
            loss = criterion(output, target, None)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['vis_global_steps']

                    idx = np.random.randint(0, num_images)

                    input_image = input.detach().cpu().numpy()[idx]
                    min_val = input_image.min()
                    max_val = input_image.max()
                    input_image = (input_image - min_val) / (max_val - min_val)
                    heatmap_target = target.detach().cpu().numpy()[idx]
                    heatmap_pred = output.detach().cpu().numpy()[idx]
                    heatmap_pred[heatmap_pred < 0.0] = 0
                    heatmap_pred[heatmap_pred > 1.0] = 1.0

                    writer.add_image('input_recording', input_image, global_steps,
                        dataformats='CHW')
                    writer.add_image('heatmap_target', heatmap_target, global_steps,
                        dataformats='CHW')
                    writer.add_image('heatmap_pred', heatmap_pred, global_steps,
                        dataformats='CHW')

                    writer_dict['vis_global_steps'] = global_steps + 1

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)

        perf_indicator = losses.avg

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
