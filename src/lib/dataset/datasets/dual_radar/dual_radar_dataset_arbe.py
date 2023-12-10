import copy
import pickle

import numpy as np
from skimage import io

import yaml
from pathlib import Path
from easydict import EasyDict

from .ops.roiaware_pool3d import roiaware_pool3d_utils
from .utils import box_utils, calibration_dual_radar, common_utils, object3d_dual_radar
# from ..dataset import DatasetTemplate
from ...generic_dataset import GenericDataset
from utils.image import get_affine_transform, affine_transform

import cv2

import glob

class DualradarDataset_ARBE(GenericDataset):
    default_resolution = [1440, 1920]
    num_categories = 3
    class_name = ['Car', 'Pedestrian', 'Cyclist']
    cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    max_objs = 128

    def __init__(self, opt, split):
        img_dir = "/root/code/CenterFusion/data/dual_radar/training/image/"
        ann_path = "/root/code/CenterFusion/data/dual_radar/training/label/"
        pc_path = "/root/code/CenterFusion/data/dual_radar/training/arbe/"
        calib_path = "/root/code/CenterFusion/data/dual_radar/training/calib/"
        self.num_categories = opt.num_classes
        self.class_name = ['' for _ in range(self.num_categories)]
        self.default_resolution = [opt.input_h, opt.input_w]
        self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}

        self.images = None
        # load image list and coco
        super().__init__(opt, split, ann_path, img_dir)

        self.root_path = Path(
            "/media/charles/ShareDisk/00myDataSet/Dual_radar/")
        self.split = split
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')
        
        data_format = '.png'
        self.img_file_list = glob.glob(str(Path(img_dir) / f'*{data_format}')) if Path(img_dir).is_dir() else [Path(img_dir)]
        data_format = '.bin'
        self.pc_file_list = glob.glob(str(Path(pc_path) / f'*{data_format}')) if Path(pc_path).is_dir() else [Path(pc_path)]
        data_format = '.txt'
        self.label_file_list = glob.glob(str(Path(ann_path) / f'*{data_format}')) if Path(ann_path).is_dir() else [Path(ann_path)]
        data_format = '.txt'
        self.calib_file_list = glob.glob(str(Path(calib_path) / f'*{data_format}')) if Path(calib_path).is_dir() else [Path(calib_path)]
    
    def __len__(self):
        print('Loaded {} samples'.format(len(self.img_file_list)))
        return len(self.img_file_list)  
        
    def __getitem__(self, index):
        opt = self.opt
        radar_pc = self.get_arbe(index)
        img = self.get_image(index)
        anns = self.get_label(index)
        calib_ = self.get_calib(index)
        P2 = np.concatenate([calib_.P2, np.array([[0., 0., 0., 1.]])], axis=0)

        height, width = img.shape[0], img.shape[1]

        img_info = {'width': None, 'height': None, 'camera_intrinsic': None, 'calib': None, 'radar_pc': None}
        img_info['width'] = width
        img_info['height'] = height
        img_info['camera_intrinsic'] = P2
        img_info['calib'] = P2[:3, :3]
        img_info['radar_pc'] = radar_pc

        # Get center and scale from image
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
            else np.array([img.shape[1], img.shape[0]], np.float32)

        aug_s, rot, flipped = 1, 0, 0

        # data augmentation for training set
        if 'train' in self.split:
            c, aug_s, rot = self._get_aug_param(c, s, width, height)  # 数据增强
            s = s * aug_s
            if np.random.random() < opt.flip:
                flipped = 1
                img = img[:, ::-1, :]
                anns = self._flip_anns(anns, width)

        # 输入输出的仿射变换矩阵
        trans_input = get_affine_transform(
            c, s, rot, [opt.input_w, opt.input_h])
        trans_output = get_affine_transform(
            c, s, rot, [opt.output_w, opt.output_h])
        inp = self._get_input(img, trans_input)  # 获取图像数据
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

        # load point cloud data
        # 加载毫米波点云数据
        if opt.pointcloud:
            pc_2d, pc_N, pc_dep, pc_3d = self._load_pc_data(
                img, img_info, trans_input, trans_output, flipped)
            ret.update({'pc_2d': pc_2d,
                        'pc_3d': pc_3d,
                        'pc_N': pc_N,
                        'pc_dep': pc_dep})
        pre_cts, track_ids = None, None

        # init samples
        self._init_ret(ret, gt_det)
        calib = self._get_calib(img_info, width, height)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            
            if ann['category_id'] < 0 or ann['category_id'] > self.opt.num_classes:
                continue
            
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue

            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)

            # 构造最终实例
            self._add_instance(ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
                               calib, pre_cts, track_ids)
            
        ret['calib'] = calib

        return ret

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(
            split_dir).readlines()] if split_dir.exists() else None

    def get_arbe(self, idx):
        arbe_file = self.pc_file_list[idx]
        # 修改1  点数变为5
        points_arbe = np.fromfile(
            str(arbe_file), dtype=np.float32).reshape(-1, 5)
                
        num_point = points_arbe.shape[0]
        points_arbe_hom = np.hstack(
            (points_arbe[:, 0:3], np.ones((num_point, 1))))
        ARB2V = np.array([0.9981128011677526, 0.05916115557244023, 0.016455814060541557, 0.07800451346438697, -0.059503891609836816, 0.9980033119043885, -
                         0.021181980812864695, 2.214080041485726, 0.015169806470300943, 0.022121191179620064, 0.9996402002082792, -1.6030740415943632]).reshape([3, 4])
        point_lidar = np.dot(points_arbe_hom, np.transpose(ARB2V))
        points_arbe[:, 0:3] = point_lidar
        
        points_arbe = points_arbe.transpose()
        return points_arbe

    def get_image(self, idx):
        img_file = self.img_file_list[idx]
        img = cv2.imread(img_file)
        cv2.resize(img,(1440,1920))
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img

    def get_label(self, idx):
        label_file = self.label_file_list[idx]
        return object3d_dual_radar.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.calib_file_list[idx]
        return calibration_dual_radar.Calibration(calib_file)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(
            pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(
            pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            # 修改5  将特征数4 改成6
            pc_info = {'num_features': 5, 'arbe_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx,
                          'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate(
                [calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate(
                [calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4,
                          'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array(
                    [obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array(
                    [obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array(
                    [obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array(
                    [obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate(
                    [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array(
                    [[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate(
                    [obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array(
                    [obj.ry for obj in obj_list])
                annotations['score'] = np.array(
                    [obj.score for obj in obj_list])
                annotations['difficulty'] = np.array(
                    [obj.level for obj in obj_list], np.int32)
                # 修改6 增加track_id
                annotations['track_id'] = np.array(
                    [obj.track_id for obj in obj_list], np.int32)

                num_objects = len(
                    [obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + \
                    [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # kitti相加坐标系以地面为中心 lidar坐标系通常以中心点为中心
                loc_lidar[:, 2] += h[:, 0] / 2
                # 修改7  因为我们的数据激光坐标系以y轴为前方 而kitti坐标系以x轴为前方 所以这里不用加np.pi / 2
                gt_boxes_lidar = np.concatenate(
                    [loc_lidar, l, w, h, -(rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_arbe'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_arbe(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(
                        pts_rect, info['image']['image_shape'], calib)
                    # 这里是取相机视角范围内的点
                    pts_fov = points[fov_flag]
                    # 返回激光坐标系下的3D框
                    corners_lidar = box_utils.boxes_to_corners_3d(
                        gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(
                            pts_fov[:, 0:3], corners_lidar[k])
                        # 计算标注信息3D框中点个数
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # 多线程 此处4个线程 对一批编号sample_id_list的数据用同一个函数process_single_scene分四个线程执行
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    # 从pkl文件中读取
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(
            self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # 做数据增强是读取这里面的文件
        db_info_save_path = Path(self.root_path) / \
            ('dual_radar_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        # 打开pkl文件
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['arbe_idx']
            points = self.get_arbe(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_arbe']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints) 返回每帧数据中每个box对应的点的索引
            # 对每个3dbox存储bbox内的点
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                # 通过索引获取到相应3dbox内的点
                gt_points = points[point_indices[i] > 0]
                # 减去3dbox中心位置坐标 得到统一坐标系下的点位置
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # gt_database/xxxxx.bin
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_arbe': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    # 预测输出的是boxes为激光坐标系下的boxes
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_arbe': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, calib)
            # 将camera下的3Dbox转化为图像的2Dbox
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1],
                                             pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_arbe'] = pred_boxes

            return pred_dict

        # batch_dict:frame_id:
        annos = []
        """
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
        """
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            # 若有输出路径  则将预测的label输出到output_path
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        # 向文件以kitti格式输出 dimensions以hwl输出
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos
