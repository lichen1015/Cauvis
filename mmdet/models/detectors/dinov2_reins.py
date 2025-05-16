from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
from mmdet.structures import OptSampleList, SampleList
from mmdet.registry import MODELS
from .dino import DINO
from ..backbones.reins_utils import set_requires_grad, set_train


@MODELS.register_module()
class ReinsDINO(DINO):
    def __init__(self, *args, replace_query_feat=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # del self.query_embedding
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            self.querys2feat = nn.Linear(self.embed_dims, self.embed_dims)

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x, querys = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return [x, querys]


    def forward_transformer(self,
                            img_feats,
                            batch_data_samples: OptSampleList = None) -> Dict:
        img_feats, query_embeddings = img_feats
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples,
            query_embeddings=query_embeddings)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(  # TODO reins add query embeddings
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None,
            query_embeddings: Tensor = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                      self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        if self.replace_query_feat:
            if query_embeddings.ndim == 2:
                query_embeddings = query_embeddings.expand(bs, -1, -1)
            query = self.querys2feat(query_embeddings)
        else:
            query = self.query_embedding.weight[:, None, :]
            query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
