import torch
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class DefDetrDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, freeze_img,
                 img_backbone,
                 img_neck,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_neck,
                 pts_bbox_head,
                 train_cfg,
                 test_cfg):
        super(DefDetrDetector, self).__init__(img_backbone=img_backbone,
                                              img_neck=img_neck,
                                              pts_voxel_layer=pts_voxel_layer,
                                              pts_voxel_encoder=pts_voxel_encoder,
                                              pts_middle_encoder=pts_middle_encoder,
                                              pts_backbone=pts_backbone,
                                              pts_neck=pts_neck,
                                              pts_bbox_head=pts_bbox_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg)
        self.freeze_img = freeze_img

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses
