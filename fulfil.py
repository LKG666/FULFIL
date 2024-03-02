"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import os
import pdb
import math
import copy
from collections import defaultdict, OrderedDict
from pathlib import Path
from functools import partial
from typing import Callable, Hashable, Sequence, Dict, Any, Type

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as torchfun
import numpy as np
from loguru import logger
from torchvision.models.detection.rpn import AnchorGenerator

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

from nndet.utils.tensor import to_numpy
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.seg import SegmentationEvaluator

from nndet.core.retina import BaseRetinaNet
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.boxes.coder import CoderType, BoxCoderND
from nndet.core.boxes.anchors import get_anchor_generator
from nndet.core.boxes.ops import box_iou, box_iou_union_3d
from nndet.core.boxes.anchors import AnchorGeneratorType

from nndet.ptmodule.base_module import LightningBaseModuleSWA, LightningBaseModule

from nndet.arch.conv import Generator, ConvInstanceRelu, ConvGroupRelu
from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.encoder.modular import Encoder
from nndet.arch.decoder.base import DecoderType, BaseUFPN, UFPNModular
from nndet.arch.heads.classifier import ClassifierType, CEClassifier
from nndet.arch.heads.regressor import RegressorType, L1Regressor
from nndet.arch.heads.comb import HeadType, DetectionHeadHNM
from nndet.arch.heads.segmenter import SegmenterType, DiCESegmenter

from nndet.training.optimizer import get_params_no_wd_on_norm
from nndet.training.learning_rate import LinearWarmupPolyLR

from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import BoxSweeper
from nndet.inference.transforms import get_tta_transforms, Inference2D
from nndet.inference.loading import get_loader_fn
from nndet.inference.helper import predict_dir
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.ensembler.detection import BoxEnsemblerSelective

from nndet.io.transforms import (
    Compose,
    Instances2Boxes,
    Instances2Segmentation,
    FindInstances,
    )

from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN_Trainer(nn.Module):
    def __init__(self, gcn_m):
        super(GCN_Trainer, self).__init__()
        self.gcn_m = gcn_m
        self.optim = torch.optim.Adam(self.gcn_m.parameters(),
                       lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        x = torch.arange(400)
        y = torch.arange(400)
        gridx, gridy = torch.meshgrid(x, y)
        xy = torch.cat([gridx[:,:, None], gridy[:,:,None]], dim=2 )
        xy = xy.flatten(0,1)
        self.position = xy
        self.s = 0

    def cal_l(self, fea):
        adj = torch.mm(fea, fea.t()).detach().cuda()
        adj = adj / adj.max()# (adj.max(dim=1)[0].view((400,1)) + 1e-8)
        edge = adj > 0.4
        adj = adj.flatten()
        edge = edge.flatten()
        edge_weight = adj[edge].cuda().float()
        edge_index = self.position[edge].t().cuda().long()
        return edge_index, edge_weight

    def reset_para(self):
        self.gcn_m.conv1.reset_parameters()
        self.gcn_m.conv2.reset_parameters()
        return

    def train_epoch(self, fea, train_mask, train_labels):
        self.gcn_m.train()
        labels = train_labels.cuda().long()
        fea = fea.detach()
        edge_index, edge_weight = self.cal_l(fea)
        e = 20
        acc = 0
        for i in range(e):
            self.optim.zero_grad()
            output = self.gcn_m(fea, edge_index, edge_weight)
            loss = self.criterion(output[train_mask], labels[train_mask])
            acc0 = (torch.argmax(output[train_mask], dim=1) == labels[train_mask]).sum() / train_mask.sum()
            acc += acc0
            if acc0 == 1:
                #print(i)
                break
            loss.backward()
            self.optim.step()
        return

    def fit(self, fea, eval_mask):
        self.gcn_m.eval()
        fea = fea.detach()
        edge_index, edge_weight = self.cal_l(fea)
        with torch.no_grad():
            output = self.gcn_m(fea, edge_index, edge_weight)
        output = torch.softmax(output, dim=1)
        return torch.argmax(output, dim=1), output[:, 1]

class GCN_TSModule(LightningBaseModuleSWA):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu
    block = StackedConvBlock2
    encoder_cls = Encoder
    decoder_cls = UFPNModular
    matcher_cls = IoUMatcher
    head_cls = DetectionHeadHNM
    head_classifier_cls = CEClassifier
    head_regressor_cls = L1Regressor
    head_sampler_cls = HardNegativeSamplerBatched
    segmenter_cls = DiCESegmenter

    def __init__(self,
                 model_cfg: dict,
                 trainer_cfg: dict,
                 plan: dict,
                 **kwargs
                 ):
        """
        RetinaUNet Lightning Module Skeleton

        Args:
            model_cfg: model configuration. Check :method:`from_config_plan`
                for more information
            trainer_cfg: trainer information
            plan: contains parameters which were derived from the planning
                stage
        """
        super().__init__(
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            plan=plan,
        )
        self.gcn_model = GCN()
        self.gcn = GCN_Trainer(self.gcn_model)
        self.model_t = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
        )
        self.warm_up_steps = 4000
        self.current_step = 0

        _classes = [f"class{c}" for c in range(plan["architecture"]["classifier_classes"])]
        self.box_evaluator = BoxEvaluator.create(
            classes=_classes,
            fast=True,
            save_dir=None,
            )
        self.seg_evaluator = SegmentationEvaluator.create()

        self.pre_trafo = Compose(
            FindInstances(
                instance_key="target",
                save_key="present_instances",
                ),
            Instances2Boxes(
                instance_key="target",
                map_key="instance_mapping",
                box_key="boxes",
                class_key="classes",
                present_instances="present_instances",
                ),
            Instances2Segmentation(
                instance_key="target",
                map_key="instance_mapping",
                present_instances="present_instances",
                )
            )

        self.eval_score_key = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"

    def normal_training_step(self, batch, batch_idx, batch_w=None):
        losses, _, f2, gcn_data = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch['target'][:, 0]  # Remove channel dimension
                },
            evaluation=False,
            batch_num=batch_idx,
        )
        if batch_w is not None:
            losses1, _, f1, gcn_data = self.model.train_step(
                    images=batch_w["data"],
                    targets={
                        "target_boxes": batch["boxes"],
                        "target_classes": batch["classes"],
                        "target_seg": batch['target'][:, 0]  # Remove channel dimension
                        },
                    evaluation=False,
                    batch_num=batch_idx,
                    )
            kl_loss = 0
            for ind in [-1,-2,-3]:
                f11 = torchfun.log_softmax(f1[0][ind], dim=1)
                f22 = torchfun.log_softmax(f2[0][ind], dim=1)
                kl_loss = kl_loss + 0.1 * (torchfun.kl_div(f11, f22, log_target=True) + torchfun.kl_div(f22, f11, log_target=True))
            loss = sum(losses.values()) + kl_loss 
            return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}
        loss = sum(losses.values())

        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def _update_teacher_model(self, keep_rate=0.996):

        student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_t.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_t.load_state_dict(new_teacher_dict)

    def merge_unlabel_label(self, batch, unlabel_targets):
        features, ind_pos, ind_neg, ind_test = [], [], [], []
        boxes_per_image = []
        th = 0.9 -  (self.current_step - 4000) / (60 * 2500 - 4000) * 0.2
        for i in range(len(batch["data"])):
            raw_boxes = batch["boxes"][i]
            raw_classes = batch["classes"][i]
            features.append(unlabel_targets["features"][i])
            boxes_per_image.append(len(features[-1]))
            if 0 in raw_boxes.shape:
                scores = unlabel_targets["pred_scores"][i]
                ind_test.append(scores > th)
                ind_neg.append(scores <= 0.2)
                ind_pos.append(scores > 1.1)
            else:
                iou, u = box_iou_union_3d(raw_boxes, unlabel_targets["pred_boxes"][i])
                indx = (iou < 0.3).sum(dim=0) == len(iou) # un matched
                indxx = (iou >= 0.3).sum(dim=0) > 0 # matched
                ind_pos.append(torch.logical_and(unlabel_targets["pred_scores"][i] > 0.7, indxx))
                ind_neg.append(torch.logical_and(unlabel_targets["pred_scores"][i]<= 0.2, indx))
                ind_test.append(torch.logical_and(unlabel_targets["pred_scores"][i] > th, indx))
        batch_ind_pos = torch.cat(ind_pos, dim=0)
        train_labels = batch_ind_pos
        batch_ind_neg = torch.cat(ind_neg, dim=0)
        eval_mask = torch.cat(ind_test, dim=0)
        train_mask = torch.logical_or(batch_ind_pos, batch_ind_neg)
        if train_labels.sum() < 1:
            return batch
        features = torch.cat(features, dim=0)
        self.gcn.train_epoch(features, train_mask, train_labels)
        with torch.no_grad():
            gcn_pred_cls, gcn_pred_scores = self.gcn.fit(features, eval_mask)
            eval_label = torch.logical_and(gcn_pred_cls, eval_mask)
            eval_label = torch.split(eval_label, boxes_per_image, dim=0)
            # pdb.set_trace()
            for i, indx in enumerate(eval_label):
                if indx.sum() == 0: continue
                raw_boxes = batch["boxes"][i]
                raw_classes = batch["classes"][i]
                pred_boxes = unlabel_targets["pred_boxes"][i][indx]
                pred_labels = unlabel_targets["pred_labels"][i][indx]
                pred_scores = unlabel_targets["pred_scores"][i][indx]
                if 0 in raw_boxes.shape:
                    new_boxes = pred_boxes
                    new_classes = pred_labels
                else:
                    new_boxes = torch.cat([raw_boxes, pred_boxes], dim=0)
                    new_classes = torch.cat([raw_classes, pred_labels], dim=0)
                #pdb.set_trace()
                #print(pred_scores)
                #print(unlabel_targets["pred_scores"][i][:10])
                #print(ind_pos[i][:10])
                batch["boxes"][i] = new_boxes
                batch["classes"][i] = new_classes
            # pdb.set_trace()
        #pred_segs = unlabel_targets["pred_seg"][:,1:2]
        #batch["target"] = batch["target"] + pred_segs

        return batch

    def training_step(self, batch, batch_idx):
        """
        Computes a single training step
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            if "w" in batch:
                batch_weak = self.pre_trafo(**batch["w"])
                batch_strong = self.pre_trafo(**batch["s"])
            else:
                batch = self.pre_trafo(**batch)
        self.current_step += 1
        if self.current_step < self.warm_up_steps:
            return self.normal_training_step(batch=batch_strong, batch_idx=batch_idx)
        else:
            if self.current_step == self.warm_up_steps:
                self._update_teacher_model(keep_rate=0)
                print(f"update teacher model at {self.current_step}")
            elif (self.current_step - self.warm_up_steps ) % 1 == 0:
                self._update_teacher_model(keep_rate=0.996)
                if self.current_step % 2500 == 0:
                    self.gcn.reset_para()
                    print(f"update teacher model at {self.current_step}")
            with torch.no_grad():
                if "w" in batch:
                    unlabel_data = batch["w"]["data"]
                    batch = batch_strong
                else:
                    unlabel_data = batch["data"]

                unlabel_targets = self.model_t.inference_step(unlabel_data)
            new_batch = self.merge_unlabel_label(batch, unlabel_targets)
            return self.normal_training_step(batch=new_batch, batch_idx=batch_idx, batch_w=batch_weak)

    def validation_step(self, batch, batch_idx):
        """
        Computes a single validation step (same as train step but with
        additional prediciton processing)
        See :class:`BaseRetinaNet` for more information
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                    "target_boxes": batch["boxes"],
                    "target_classes": batch["classes"],
                    "target_seg": batch['target'][:, 0]  # Remove channel dimension
                }
            losses, prediction, _, _ = self.model_t.train_step(
                images=batch["data"],
                targets=targets,
                evaluation=True,
                batch_num=batch_idx,
            )
            loss = sum(losses.values())

        self.evaluation_step(prediction=prediction, targets=targets)
        return {"loss": loss.detach().item(),
                **{key: l.detach().item() for key, l in losses.items()}}

    def evaluation_step(
        self,
        prediction: dict,
        targets: dict,
    ):
        """
        Perform an evaluation step to add predictions and gt to
        caching mechanism which is evaluated at the end of the epoch

        Args:
            prediction: predictions obtained from model
                'pred_boxes': List[Tensor]: predicted bounding boxes for
                    each image List[[R, dim * 2]]
                'pred_scores': List[Tensor]: predicted probability for
                    the class List[[R]]
                'pred_labels': List[Tensor]: predicted class List[[R]]
                'pred_seg': Tensor: predicted segmentation [N, dims]
            targets: ground truth
                `target_boxes` (List[Tensor]): ground truth bounding boxes
                    (x1, y1, x2, y2, (z1, z2))[X, dim * 2], X= number of ground
                        truth boxes in image
                `target_classes` (List[Tensor]): ground truth class per box
                    (classes start from 0) [X], X= number of ground truth
                    boxes in image
                `target_seg` (Tensor): segmentation ground truth (if seg was
                    found in input dict)
        """
        pred_boxes = to_numpy(prediction["pred_boxes"])
        pred_classes = to_numpy(prediction["pred_labels"])
        pred_scores = to_numpy(prediction["pred_scores"])

        gt_boxes = to_numpy(targets["target_boxes"])
        gt_classes = to_numpy(targets["target_classes"])
        gt_ignore = None

        self.box_evaluator.run_online_evaluation(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            gt_ignore=gt_ignore,
            )

        #pred_seg = to_numpy(prediction["pred_seg"])
        #gt_seg = to_numpy(targets["target_seg"])

        #self.seg_evaluator.run_online_evaluation(
        #    seg_probs=pred_seg,
        #    target=gt_seg,
        #    )

    def training_epoch_end(self, training_step_outputs):
        """
        Log train loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in training_step_outputs:
            for _k, _v in _val.items():
                if _k == "loss":
                    vals[_k].append(_v.detach().item())
                else:
                    vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = np.mean(_vals)
            if _key == "loss":
                logger.info(f"Train loss reached: {mean_val:0.5f}")
            self.log(f"train_{_key}", mean_val, sync_dist=True)
        return super().training_epoch_end(training_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        """
        Log val loss to loguru logger
        """
        # process and log losses
        vals = defaultdict(list)
        for _val in validation_step_outputs:
            for _k, _v in _val.items():
                vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = np.mean(_vals)
            if _key == "loss":
                logger.info(f"Val loss reached: {mean_val:0.5f}")
            self.log(f"val_{_key}", mean_val, sync_dist=True)

        # process and log metrics
        self.evaluation_end()
        return super().validation_epoch_end(validation_step_outputs)

    def evaluation_end(self):
        """
        Uses the cached values from `evaluation_step` to perform the evaluation
        of the epoch
        """
        metric_scores, _ = self.box_evaluator.finish_online_evaluation()
        self.box_evaluator.reset()

        logger.info(f"mAP@0.1:0.5:0.05: {metric_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:0.3f}  "
                    f"AP@0.1: {metric_scores['AP_IoU_0.10_MaxDet_100']:0.3f}  "
                    f"AP@0.5: {metric_scores['AP_IoU_0.50_MaxDet_100']:0.3f}")

        seg_scores, _ = self.seg_evaluator.finish_online_evaluation()
        self.seg_evaluator.reset()
        metric_scores.update(seg_scores)

        #logger.info(f"Proxy FG Dice: {seg_scores['seg_dice']:0.3f}")

        for key, item in metric_scores.items():
            self.log(f'{key}', item, on_step=None, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        Base configuration is SGD with LinearWarmup and PolyLR learning rate
        schedule
        """
        # configure optimizer
        logger.info(f"Running: initial_lr {self.trainer_cfg['initial_lr']} "
                    f"weight_decay {self.trainer_cfg['weight_decay']} "
                    f"SGD with momentum {self.trainer_cfg['sgd_momentum']} and "
                    f"nesterov {self.trainer_cfg['sgd_nesterov']}")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = torch.optim.SGD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
            )

        # configure lr scheduler
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer], {'scheduler': scheduler, 'interval': 'step'}

    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        """
        Create Configurable RetinaUNet

        Args:
            model_cfg: model configurations
                See example configs for more info
            plan_arch: plan architecture
                `dim` (int): number of spatial dimensions
                `in_channels` (int): number of input channels
                `classifier_classes` (int): number of classes
                `seg_classes` (int): number of classes
                `start_channels` (int): number of start channels in encoder
                `fpn_channels` (int): number of channels to use for FPN
                `head_channels` (int): number of channels to use for head
                `decoder_levels` (int): decoder levels to user for detection
            plan_anchors: parameters for anchors (see
                :class:`AnchorGenerator` for more info)
                    `stride`: stride
                    `aspect_ratios`: aspect ratios
                    `sizes`: sized for 2d acnhors
                    (`zsizes`: additional z sizes for 3d)
            log_num_anchors: name of logger to use; if None, no logging
                will be performed
            **kwargs:
        """
        logger.info(f"Architecture overwrites: {model_cfg['plan_arch_overwrites']} "
                    f"Anchor overwrites: {model_cfg['plan_anchors_overwrites']}")
        logger.info(f"Building architecture according to plan of {plan_arch.get('arch_name', 'not_found')}")
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])
        logger.info(f"Start channels: {plan_arch['start_channels']}; "
                    f"head channels: {plan_arch['head_channels']}; "
                    f"fpn channels: {plan_arch['fpn_channels']}")

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.,) * (plan_arch["dim"] * 2))
        s_param = False if ("aspect_ratios" in _plan_anchors) and \
                           (_plan_anchors["aspect_ratios"] is not None) else True
        anchor_generator = get_anchor_generator(
            plan_arch["dim"], s_param=s_param)(**_plan_anchors)

        encoder = cls._build_encoder(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        decoder = cls._build_decoder(
            encoder=encoder,
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
            )

        classifier = cls._build_head_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        regressor = cls._build_head_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        head = cls._build_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
            coder=coder
        )
        segmenter = cls._build_segmenter(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )

        detections_per_img = plan_arch.get("detections_per_img", 100)
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
        nms_thresh = plan_arch.get("nms_thresh", 0.6)

        logger.info(f"Model Inference Summary: \n"
                    f"detections_per_img: {detections_per_img} \n"
                    f"score_thresh: {score_thresh} \n"
                    f"topk_candidates: {topk_candidates} \n"
                    f"remove_small_boxes: {remove_small_boxes} \n"
                    f"nms_thresh: {nms_thresh}",
                    )

        return BaseRetinaNet(
            dim=plan_arch["dim"],
            encoder=encoder,
            decoder=decoder,
            head=head,
            anchor_generator=anchor_generator,
            matcher=matcher,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            segmenter=segmenter,
            # model_max_instances_per_batch_element (in mdt per img, per class; here: per img)
            detections_per_img=detections_per_img,
            score_thresh=score_thresh,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
        )

    @classmethod
    def _build_encoder(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> EncoderType:
        """
        Build encoder network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            EncoderType: encoder instance
        """
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(f"Building:: encoder {cls.encoder_cls.__name__}: {model_cfg['encoder_kwargs']} ")
        encoder = cls.encoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            stage_kwargs=None,
            max_channels=plan_arch.get("max_channels", 320),
            **model_cfg['encoder_kwargs'],
        )
        return encoder

    @classmethod
    def _build_decoder(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        encoder: EncoderType,
    ) -> DecoderType:
        """
        Build decoder network

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            DecoderType: decoder instance
        """
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(f"Building:: decoder {cls.decoder_cls.__name__}: {model_cfg['decoder_kwargs']}")
        decoder = cls.decoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=encoder.get_strides(),
            in_channels=encoder.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            fixed_out_channels=plan_arch["fpn_channels"],
            **model_cfg['decoder_kwargs'],
        )
        return decoder

    @classmethod
    def _build_head_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> ClassifierType:
        """
        Build classification subnetwork for detection head

        Args:
            anchor_generator: anchor generator instance
            plan_arch: architecture settings
            model_cfg: additional architecture settings

        Returns:
            ClassifierType: classification instance
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_classifier_cls.__name__
        kwargs = model_cfg['head_classifier_kwargs']

        logger.info(f"Building:: classifier {name}: {kwargs}")
        classifier = cls.head_classifier_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            num_classes=plan_arch["classifier_classes"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return classifier

    @classmethod
    def _build_head_regressor(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> RegressorType:
        """
        Build regression subnetwork for detection head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            anchor_generator: anchor generator instance

        Returns:
            RegressorType: classification instance
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_regressor_cls.__name__
        kwargs = model_cfg['head_regressor_kwargs']

        logger.info(f"Building:: regressor {name}: {kwargs}")
        regressor = cls.head_regressor_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return regressor

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: ClassifierType,
        regressor: RegressorType,
        coder: CoderType,
    ) -> HeadType:
        """
        Build detection head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            classifier: classifier instance
            regressor: regressor instance
            coder: coder instance to encode boxes

        Returns:
            HeadType: instantiated head
        """
        head_name = cls.head_cls.__name__
        head_kwargs = model_cfg['head_kwargs']
        sampler_name = cls.head_sampler_cls.__name__
        sampler_kwargs = model_cfg['head_sampler_kwargs']

        logger.info(f"Building:: head {head_name}: {head_kwargs} "
                    f"sampler {sampler_name}: {sampler_kwargs}")
        sampler = cls.head_sampler_cls(**sampler_kwargs)
        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            sampler=sampler,
            log_num_anchors=None,
            **head_kwargs,
        )
        return head

    @classmethod
    def _build_segmenter(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        decoder: DecoderType,
    ) -> SegmenterType:
        """
        Build segmenter head

        Args:
            plan_arch: architecture settings
            model_cfg: additional architecture settings
            decoder: decoder instance

        Returns:
            SegmenterType: segmenter head
        """
        if cls.segmenter_cls is not None:
            name = cls.segmenter_cls.__name__
            kwargs = model_cfg['segmenter_kwargs']
            conv = Generator(cls.base_conv_cls, plan_arch["dim"])

            logger.info(f"Building:: segmenter {name} {kwargs}")
            segmenter = cls.segmenter_cls(
                conv,
                seg_classes=plan_arch["seg_classes"],
                in_channels=decoder.get_channels(),
                decoder_levels=plan_arch["decoder_levels"],
                **kwargs,
            )
        else:
            segmenter = None
        return segmenter

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        Get ensembler classes to combine multiple predictions
        Needs to be overwritten in subclasses!
        """
        _lookup = {
            2: {
                "boxes": None,
                "seg": None,
            },
            3: {
                "boxes": BoxEnsemblerSelective,
                "seg": SegmentationEnsembler,
                }
            }
        if dim == 2:
            raise NotImplementedError
        return _lookup[dim][key]

    @classmethod
    def get_predictor(cls,
                      plan: Dict,
                      models: Sequence[RetinaUNetModule],
                      num_tta_transforms: int = None,
                      do_seg: bool = False,
                      **kwargs,
                      ) -> Predictor:
        # process plan
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inferene_plan = plan.get("inference_plan", {})
        logger.info(f"Found inference plan: {inferene_plan} for prediction")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # setup
        tta_transforms, tta_inverse_transforms = \
            get_tta_transforms(num_tta_transforms, True)
        logger.info(f"Using {len(tta_transforms)} tta transformations for prediction (one dummy trafo).")

        ensembler = {"boxes": partial(
            cls.get_ensembler_cls(key="boxes", dim=plan["network_dim"]).from_case,
            parameters=inferene_plan,
        )}
        if do_seg:
            ensembler["seg"] = partial(
                cls.get_ensembler_cls(key="seg", dim=plan["network_dim"]).from_case,
            )

        predictor = Predictor(
            ensembler=ensembler,
            models=models,
            crop_size=crop_size,
            tta_transforms=tta_transforms,
            tta_inverse_transforms=tta_inverse_transforms,
            batch_size=batch_size,
            **kwargs,
            )
        if plan["network_dim"] == 2:
            raise NotImplementedError
            predictor.pre_transform = Inference2D(["data"])
        return predictor

    def sweep(self,
              cfg: dict,
              save_dir: os.PathLike,
              train_data_dir: os.PathLike,
              case_ids: Sequence[str],
              run_prediction: bool = True,
              **kwargs,
              ) -> Dict[str, Any]:
        """
        Sweep detection parameters to find the best predictions

        Args:
            cfg: config used for training
            save_dir: save dir used for training
            train_data_dir: directory where preprocessed training/validation
                data is located
            case_ids: case identifies to prepare and predict
            run_prediction: predict cases
            **kwargs: keyword arguments passed to predict function

        Returns:
            Dict: inference plan
                e.g. (exact params depend on ensembler class usef for prediction)
                `iou_thresh` (float): best IoU threshold
                `score_thresh (float)`: best score threshold
                `no_overlap` (bool): enable/disable class independent NMS (ciNMS)
        """
        logger.info(f"Running parameter sweep on {case_ids}")

        train_data_dir = Path(train_data_dir)
        preprocessed_dir = train_data_dir.parent
        processed_eval_labels = preprocessed_dir / "labelsTr"

        _save_dir = save_dir / "sweep"
        _save_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = save_dir / "sweep_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        if run_prediction:
            logger.info("Predict cases with default settings...")
            predictor = predict_dir(
                source_dir=train_data_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=self.plan,
                source_models=save_dir,
                num_models=1,
                num_tta_transforms=1,
                case_ids=case_ids,
                save_state=True,
                model_fn=get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "last")),
                **kwargs,
                )

        logger.info("Start parameter sweep...")
        ensembler_cls = self.get_ensembler_cls(key="boxes", dim=self.plan["network_dim"])
        sweeper = BoxSweeper(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=prediction_dir,
            gt_dir=processed_eval_labels,
            target_metric=self.eval_score_key,
            ensembler_cls=ensembler_cls,
            save_dir=_save_dir,
            )
        inference_plan = sweeper.run_postprocessing_sweep()
        return inference_plan
