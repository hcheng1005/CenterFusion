from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector import Detector


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt
    self.dataset = dataset
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      input_meta = {}
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib
      images[scale], meta[scale] = self.pre_process_func(
        image, scale, input_meta)
      
    ret = {'images': images, 'image': image, 'meta': meta}
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    
    # add point cloud
    if opt.pointcloud:
      assert len(opt.test_scales)==1, "Multi-scale testing not supported with pointcloud."
      scale = opt.test_scales[0]
      pc_2d, pc_N, pc_dep, pc_3d = self.dataset._load_pc_data(image, img_info, 
        meta[scale]['trans_input'], meta[scale]['trans_output'])
      ret['pc_2d'] = pc_2d
      ret['pc_N'] = pc_N
      ret['pc_dep'] = pc_dep
      ret['pc_3d'] = pc_3d

    return img_id, ret

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  if split == 'val':
    split = opt.val_split
    
  dataset = Dataset(opt, split)
  detector = Detector(opt)
    
  load_results = {}

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  
  # if opt.use_loaded_results:
  #   for img_id in data_loader.dataset.images:
  #     results[img_id] = load_results['{}'.format(img_id)]
  #   num_iters = 0
  
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    if ind >= num_iters:
      break
    # if opt.tracking and ('is_first_frame' in pre_processed_images):
    #   if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
    #     pre_processed_images['meta']['pre_dets'] = \
    #       load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
    #   else:
    #     print()
    #     print('No pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
    #       '. Use empty initialization.')
    #     pre_processed_images['meta']['pre_dets'] = []
    #   detector.reset_tracking()
    #   print('Start tracking video', int(pre_processed_images['video_id']))
    # if opt.public_det:
    #   if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
    #     pre_processed_images['meta']['cur_dets'] = \
    #       load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
    #   else:
    #     print('No cur_dets for', int(img_id.numpy().astype(np.int32)[0]))
    #     pre_processed_images['meta']['cur_dets'] = []
    
    # print(pre_processed_images.keys())
    # dict_keys(['images', 'image', 'meta', 'is_first_frame', 'video_id', 'pc_2d', 'pc_N', 'pc_dep', 'pc_3d'])


    img_ = pre_processed_images['image'].cpu().numpy().squeeze()
    print(img_.shape)
    # cv2.imshow('img', img_)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # 图片的分辨率为200*300，这里b, g, r设为随机值，注意dtype属性
    # b = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    # g = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    # r = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

    # # 合并通道，形成图片
    # img = cv2.merge([b, g, r])

    # # 显示图片
    # cv2.imshow('test', img)
    # cv2.waitKey(1000)
    # cv2.destroyWindow('test')
  
    ret = detector.run(pre_processed_images)
    results[int(img_id.numpy().astype(np.int32)[0])] = ret['results']
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
      
    if opt.print_iter > 0:
      if ind % opt.print_iter == 0:
        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
    else:
      bar.next()
      
  bar.finish()
  
  # if opt.save_results:
  #   print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
  #     opt.test_dataset, opt.dataset_version))
  #   json.dump(_to_list(copy.deepcopy(results)), 
  #             open(opt.save_dir + '/save_results_{}{}.json'.format(
  #               opt.test_dataset, opt.dataset_version), 'w'))
  # dataset.run_eval(results, opt.save_dir, n_plots=opt.eval_n_plots, 
  #                  render_curves=opt.eval_render_curves)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().parse()
  prefetch_test(opt)
