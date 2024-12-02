import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from segment_anything.utils.amg import (
    batch_iterator, 
)

from scipy.sparse import csgraph


class GFSAM:
    def __init__(
            self,
            encoder,
            generator=None,
            input_size=1024,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # models
        self.encoder = encoder
        self.predictor = generator

        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.input_size = input_size

        img_size = 518
        feat_size = img_size // self.encoder.patch_size

        self.encoder_img_size = img_size
        self.encoder_feat_size = feat_size

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.nshot = None

        self.device = device
        
    def set_reference(self, imgs, masks):

        def reference_masks_verification(masks):
            if masks.sum() == 0:
                _, _, sh, sw = masks.shape
                masks[..., (sh // 2 - 7):(sh // 2 + 7), (sw // 2 - 7):(sw // 2 + 7)] = 1
            return masks

        imgs = imgs.flatten(0, 1)  # bs, 3, h, w

        # process reference masks
        masks = reference_masks_verification(masks)
        masks = masks.permute(1, 0, 2, 3)  # ns, 1, h, w
        nshot = masks.shape[0]

        self.ref_imgs = imgs
        self.nshot = nshot
        self.ref_masks = masks

    def set_target(self, img):

        img_h, img_w = img.shape[-2:]
        assert img_h == self.input_size[0] and img_w == self.input_size[1]

        # transform query to numpy as input of sam
        img_np = img.mul(255).byte()
        img_np = img_np.squeeze(0).permute(1, 2, 0).cpu().numpy()

        self.tar_img = img
        self.tar_img_np = img_np

    def predict(self):

        tar_feats = self.extract_sam_feats()
        ref_feats_sem, tar_feats_sem = self.extract_img_feats()

        # positive and negative similarity maps
        neg_sim_map, neg_mean_sim_map = self.generate_prior(tar_feats_sem, ref_feats_sem, 1-self.ref_masks)
        sim_map, mean_sim_map = self.generate_prior(tar_feats_sem, ref_feats_sem, self.ref_masks)

        # mid-value of similarity map
        mean_sim_map_half = (mean_sim_map.max() + mean_sim_map.min()) / 2

        # mix the similarity map and align the value to [0, 1]
        cross_sim_map = mean_sim_map * sim_map
        cross_sim_map = (cross_sim_map - cross_sim_map.min()) / (cross_sim_map.max() - cross_sim_map.min() + 1e-6)

        neg_mean_sim_map_std = (neg_mean_sim_map - neg_mean_sim_map.min()) / (neg_mean_sim_map.max() - neg_mean_sim_map.min() + 1e-6)
        neg_region = neg_mean_sim_map_std > cross_sim_map
        mean_sim_map_fil = cross_sim_map * ~neg_region
        coord_xy, coord_labels, sim_map_hot, coord_f = self.find_points(mean_sim_map_fil, tar_feats)

        tar_masks_list = self.generate_sam_masks(tar_feats, coord_xy, coord_labels)

        if len(tar_masks_list) == 0:
            tar_masks = torch.zeros((1, 1, 1024, 1024), device=self.device)
            pred_masks = tar_masks.squeeze(0)
            prob_masks = torch.zeros_like(pred_masks)
        else:
            tar_masks = torch.cat(tar_masks_list, dim=0)

            components_weak, labels_weak, components_strong, labels_strong = self.mask_cluster(tar_masks, coord_f, sim_map_hot)

            fgbg_com_labels, fgbg_labels, pseudo_masks, cls_scores = self.cluster_classification(tar_masks, labels_weak, components_weak, 
                                                                                     mean_sim_map * mean_sim_map, neg_mean_sim_map * mean_sim_map_half, coord_f)

            selected_points = self.point_consistency_dis(tar_feats_sem, tar_masks, labels_weak, components_weak, fgbg_labels, coord_f)

            pred_masks, prob_masks = self.triplet_selection_b(tar_masks, labels_weak, selected_points, mean_sim_map, coord_f, cls_scores)

        return pred_masks, (coord_xy, selected_points)
    
    def triplet_selection_b(self, tar_masks, cluster_labels, coord_se_labels, mean_sim_map, coord_f, cls_scores):
        """Select and merge the masks"""
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)
        
        pred_masks = torch.zeros(self.input_size, device=self.device).unsqueeze(0)
        prob_masks = torch.zeros(self.input_size, device=self.device).unsqueeze(0)
        sim_map_rsz = F.interpolate(mean_sim_map.unsqueeze(0), self.input_size, mode='bilinear', align_corners=False).squeeze(0)
        for idx, coord_se_label in enumerate(coord_se_labels):
            if coord_se_label == 1:
                pred_masks += tar_masks[idx]
                curr_prob_mask = (sim_map_rsz * tar_masks[idx]).sum() / tar_masks[idx].sum() * tar_masks[idx]
                prob_masks = torch.max(prob_masks, curr_prob_mask)
        pred_masks = (pred_masks > 0).float()

        if pred_masks.sum() == 0:
            cls_scores[coord_se_labels != 2] = 0
            _, max_cls_scores_arg = cls_scores.max(dim=0)
            pred_masks += tar_masks[max_cls_scores_arg]
            prob_masks = (sim_map_rsz * tar_masks[max_cls_scores_arg]).sum() / tar_masks[max_cls_scores_arg].sum() * tar_masks[max_cls_scores_arg]

        return pred_masks, prob_masks

    
    def cluster_classification(self, tar_masks, cluster_labels, n_components, mean_sim_map, neg_map, coord_f):
        """Classify each points with the guidance from cluster labels and similarity maps"""
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)

        fgbg_labels = torch.zeros_like(cluster_labels)
        fgbg_com_labels = torch.zeros(n_components, device=self.device, dtype=torch.long)
        cls_scores = torch.zeros_like(cluster_labels, dtype=torch.float)
        pos_region = (mean_sim_map > neg_map).float()
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            union_mask = (tar_masks[com_args].sum(dim=0) > 0).float()
            union_mask = F.interpolate(union_mask.unsqueeze(0), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False).squeeze(0)
            pos_score = (pos_region * union_mask).sum()
            neg_score = ((1 - pos_region) * union_mask).sum()

            fgbg_com_labels[component] = 1
            tar_mask_rsz = F.interpolate(tar_masks[com_args].float(), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False)
            mask_scores = (tar_mask_rsz * (mean_sim_map - neg_map).unsqueeze(0)).sum(dim=(-1, -2, -3))
            mask_scores = mask_scores / (tar_mask_rsz.sum(dim=(-1, -2, -3)) + 1e-6)
            mask_args = mask_scores.argsort(descending=True)
            inner_mask = torch.zeros_like(union_mask, device=self.device, dtype=torch.float)
            # Mask Growth
            for idx in mask_args:
                curr_mask = tar_mask_rsz[idx]
                curr_mask = curr_mask * (1 - inner_mask)
                pos_score = (pos_region * curr_mask).sum()
                neg_score = ((1 - pos_region) * curr_mask).sum()
                if pos_score > neg_score:
                    inner_mask += curr_mask
                    fgbg_labels[com_args[idx]] = 1
            cls_scores[com_args] = pos_score - neg_score

        pseudo_masks = (tar_masks[fgbg_labels == 1].sum(dim=0) > 0).float()

        return fgbg_com_labels, fgbg_labels, pseudo_masks, cls_scores
             
    def point_consistency_dis(self, tar_feats_sem, tar_masks, cluster_labels, n_components, fgbg_labels, coord_f):
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        cluster_labels = torch.as_tensor(cluster_labels, device=self.device, dtype=torch.long)
        fgbg_labels = torch.as_tensor(fgbg_labels, device=self.device, dtype=torch.long)

        union_masks = []
        union_similarities = []
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            union_mask = (tar_masks[com_args].sum(dim=0) > 0).float()
            union_mask = F.interpolate(union_mask.unsqueeze(0), (self.encoder_feat_size, self.encoder_feat_size), mode='bilinear', align_corners=False).squeeze(0)
            union_masks.append(union_mask)
            _, union_similarity = self.generate_prior(tar_feats_sem, tar_feats_sem, union_mask.unsqueeze(0))
            union_similarities.append(union_similarity)
        union_similarities = torch.cat(union_similarities, dim=0) # nc, h, w

        # get similarities from coordinates
        coord_similarities = union_similarities[:, coord_f[:, 1], coord_f[:, 0]] # nc, nm

        # compute distance to the nearest point in the cluster
        coord_distance = torch.cdist(coord_f.float(), coord_f.float()) # nm, nm
        coord_distance_labels = torch.zeros_like(coord_similarities)
        for idx in range(coord_f.shape[0]):
            for compo in range(n_components):
                com_args = torch.where(cluster_labels == compo)[0]
                coord_distance_labels[compo, idx] = coord_distance[idx, com_args].min()
        coord_distance_labels[coord_distance_labels == 0] = 1
        coord_similarities = coord_similarities / coord_distance_labels

        coord_similarities_max, coord_similarities_max_args = coord_similarities.max(dim=0) # nm

        # labeling the points for self consistency
        coord_selection_labels = torch.where(coord_similarities_max_args == cluster_labels, 1, 0)
        for component in range(n_components):
            com_args = torch.where(cluster_labels == component)[0]
            const_count = coord_selection_labels[com_args].sum()
            if const_count < len(com_args) - const_count:
                coord_selection_labels[com_args] = 0
        coord_pos_labels = (coord_selection_labels * fgbg_labels) > 0
        coord_waiting_labels = ((coord_selection_labels + fgbg_labels) > 0).long() * 2
        coord_waiting_labels[coord_pos_labels] = 1

        return coord_waiting_labels

    def mask_cluster(self, tar_masks, coord_f, sim_map_hot):
        """Mask clustering using connected components"""
        sim_map_hot = torch.as_tensor(sim_map_hot>0, device=self.device, dtype=torch.float)

        tar_masks_rsz = F.interpolate(tar_masks.float(), (sim_map_hot.shape[-2], sim_map_hot.shape[-1]), mode='nearest').squeeze(1)
        adjacent = tar_masks_rsz * sim_map_hot.unsqueeze(0) # Compute the coverage

        # squeeze adjacent matrix according to the coordinate
        adjacent = adjacent.flatten(1) # nm, h*w
        coord_f = torch.as_tensor(coord_f, device=self.device, dtype=torch.long)
        coord_f = coord_f[:, 1] * sim_map_hot.shape[-1] + coord_f[:, 0]
        adjacent = adjacent[:, coord_f] # nm, nm

        # strong connected components
        n_components_strong, labels_strong = csgraph.connected_components(adjacent.cpu().numpy(), directed=True, connection='strong', return_labels=True)
        n_components_weak, labels_weak = csgraph.connected_components(adjacent.cpu().numpy(), directed=True, connection='weak', return_labels=True)

        # return n_components, labels
        return n_components_weak, labels_weak, n_components_strong, labels_strong

    def find_points(self, sim_map, tar_feats):
        """Select points for prompting"""
        
        sum_sim = sim_map.sum()
        topk = min(int(sum_sim), 128) # set maximum to 128 for efficiency
        if topk == 0 and sum_sim > 0:
            topk = 1

        sim_map_flt = sim_map.flatten(0)
        sim_map_topk_args = sim_map_flt.topk(topk)[1]
        sim_map_hot = torch.zeros_like(sim_map_flt)
        sim_map_hot[sim_map_topk_args] = 1
        sim_map_hot = sim_map_hot.view(1, sim_map.shape[1], sim_map.shape[2])
        sim_map_hot = sim_map_hot.squeeze(0).cpu().numpy()

        # translate all points to coordinates
        points_f = np.argwhere(sim_map_hot.T > 0)
        points = self.predictor.transform.apply_coords(points_f, sim_map.shape[-2:])
        coord_labels = np.ones(points.shape[0], dtype=np.int32)

        return points, coord_labels, sim_map_hot, points_f

    def generate_sam_masks(self, tar_feats, coord_xy, coord_labels):
        """Generate masks using SAM"""
        tar_masks_list = []
        for (points,), (labels,) in zip(batch_iterator(64, coord_xy), batch_iterator(64, coord_labels)):
            in_points = torch.as_tensor(points, device=self.device, dtype=torch.int)
            in_labels = torch.as_tensor(labels, device=self.device, dtype=torch.int)

            tar_masks, scores, logits, _ = self.predictor.predict_torch(
                point_coords=in_points[:, None, :],
                point_labels=in_labels[:, None],
                # mask_input=mask_inputs,
                features=tar_feats,
                multimask_output=False, 
            )
            tar_masks = tar_masks > self.predictor.model.mask_threshold
            tar_masks_list.append(tar_masks)
        return tar_masks_list

    def generate_prior(self, query_feat_high, supp_feat_high, s_mask):
        bsize, sp_sz2, _= query_feat_high.size()[:]
        sp_sz = int(math.sqrt(sp_sz2))

        similarities = self.generate_pixelwise_comparison(query_feat_high, supp_feat_high)
        corr_query_mask_list = []
        cos_similarity_list = []
        cosine_eps = 1e-7
        for st, supp_feat in enumerate(supp_feat_high):
            tmp_mask = s_mask[st].unsqueeze(0)
            tmp_mask = F.interpolate(tmp_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)
            tmp_mask = tmp_mask.flatten(2).transpose(-1, -2) # [bs, h*w, 1]

            similarity = similarities[st] * tmp_mask # [bs, h*w, h*w]
            cos_similarity = similarity.mean(1).view(bsize, 1, sp_sz, sp_sz) / (tmp_mask.sum() / sp_sz2 + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz2)   
            # similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            # corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
            cos_similarity_list.append(cos_similarity)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = corr_query_mask.mean(1)
        cos_similarity = torch.cat(cos_similarity_list, 1).mean(1)
        return corr_query_mask, cos_similarity

    def generate_pixelwise_comparison(self, query_feat_high, supp_feat_high):
        pixelwise_coms = []
        for st, supp_feat in enumerate(supp_feat_high):
            tmp_supp_feat = supp_feat.unsqueeze(0)
            tmp_query = query_feat_high.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
            tmp_supp = tmp_supp_feat.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + 1e-7) # [bs, h*w, h*w]
            pixelwise_coms.append(similarity)
        return pixelwise_coms
    
    def extract_sam_feats(self):
        self.predictor.set_image(self.tar_img_np)
        tar_feats = self.predictor.features # 1, c, h, w

        return tar_feats
    
    def extract_img_feats(self):

        ref_imgs = torch.cat([self.encoder_transform(rimg)[None, ...] for rimg in self.ref_imgs], dim=0)
        tar_img = torch.cat([self.encoder_transform(timg)[None, ...] for timg in self.tar_img], dim=0)

        ref_feats = self.encoder.forward_features(ref_imgs.to(self.device))["x_prenorm"][:, 1:]
        tar_feat = self.encoder.forward_features(tar_img.to(self.device))["x_prenorm"][:, 1:]

        ref_feats = F.normalize(ref_feats, dim=1, p=2) # normalize for cosine similarity
        tar_feat = F.normalize(tar_feat, dim=1, p=2)

        return ref_feats, tar_feat
    
    def clear(self):

        self.tar_img = None
        self.tar_img_np = None

        self.ref_imgs = None
        self.nshot = None


def build_model(args):

    # DINOv2, Image Encoder
    dinov2_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2 = vits.__dict__[args.dinov2_size](**dinov2_kwargs)

    dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")
    dinov2.eval()
    dinov2.to(device=args.device)

    # SAM
    sam = sam_model_registry[args.sam_size](checkpoint=args.sam_weights)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    return GFSAM(
        encoder=dinov2,
        generator=predictor,
        device=args.device
    )
