# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
import time

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        sel_pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        sel_stability_score_thresh: float = 0.95,
        sel_stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        multimask_output: bool = True,
        sel_multimask_output: bool = True,
        output_layer: int = -1,
        sel_output_layer: int = -1,
        dense_pred: bool = True
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crops_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crops_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        self.sel_pred_iou_thresh = sel_pred_iou_thresh
        self.sel_stability_score_thresh = sel_stability_score_thresh
        self.sel_stability_score_offset = sel_stability_score_offset

        self.multimask_output = multimask_output
        self.sel_multimask_output = sel_multimask_output
        self.output_layer = output_layer
        self.sel_output_layer = sel_output_layer
        self.dense_pred = dense_pred

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        select_point_coords: Optional[List[np.ndarray]] = None,
        select_point_labels: Optional[List[np.ndarray]] = None,
        select_box: Optional[np.ndarray] = None,
        select_mask_input: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
          select_point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          select_point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          select_box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          select_mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(
            image,
            select_point_coords,
            select_point_labels,
            select_box,
            select_mask_input
        )

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                # "point_coords": [mask_data["points"][idx].tolist()],
                "low_res_masks": mask_data["low_res_masks"][idx],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(
            self,
            image: np.ndarray,
            select_point_coords: Optional[List[np.ndarray]] = None,
            select_point_labels: Optional[List[np.ndarray]] = None,
            select_box: Optional[np.ndarray] = None,
            select_mask_input: Optional[np.ndarray] = None,
    ) -> MaskData:

        orig_size = image.shape[:2]

        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            if layer_idx > 0:
                crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            else:
                crop_data = self._process_crop(
                    image,
                    crop_box,
                    layer_idx,
                    orig_size,
                    select_point_coords,
                    select_point_labels,
                    select_box,
                    select_mask_input
                )
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros(len(data["boxes"])),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        select_point_coords: Optional[List[np.ndarray]] = None,
        select_point_labels: Optional[List[np.ndarray]] = None,
        select_boxs: Optional[List[np.ndarray]] = None,
        select_masks_input: Optional[List[np.ndarray]] = None
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()

        if self.dense_pred:
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                t0 = time.time()
                batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
                t1 = time.time()
                print("Time for batch: ", t1 - t0)
                data.cat(batch_data)
                del batch_data

        if crop_layer_idx == 0 and select_point_coords is not None:

            assert select_point_labels is not None, "select_point_labels should not be None !"
            points_list_len = len(select_point_coords)

            if select_boxs == None:
                select_boxs = [None] * points_list_len
            elif isinstance(select_boxs, list) and len(select_boxs) == 1:
                select_boxs = select_boxs * points_list_len
            else:
                raise NotImplementedError

            if select_masks_input == None:
                select_masks_input = [None] * points_list_len
            elif isinstance(select_masks_input, list) and len(select_masks_input) == 1:
                select_masks_input = select_masks_input * points_list_len
            else:
                raise NotImplementedError
            for select_point_coords, select_point_labels, select_box, select_mask_input in \
                    zip(select_point_coords, select_point_labels, select_boxs, select_masks_input):
                for (sel_points, sel_labels) in batch_iterator(self.points_per_batch, select_point_coords, select_point_labels):
                    t0 = time.time()
                    batch_data = self._process_sel_batch(
                        cropped_im_size,
                        crop_box,
                        orig_size,
                        point_coords=sel_points,
                        point_labels=sel_labels,
                        boxes=select_box,
                        mask_input=select_mask_input
                    )
                    t1 = time.time()
                    print("Time for batch target: ", t1 - t0)

                    data.cat(batch_data)
                    del batch_data

        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"]), device=data["boxes"].device),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        # data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, low_res_masks, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        if self.sel_multimask_output:
            if self.output_layer in [0, 1, 2]:
                masks = masks[:, self.output_layer][:, None, :, :]
                low_res_masks = low_res_masks[:, self.output_layer][:, None, :, :]
                iou_preds = iou_preds[:, self.output_layer][:, None]

            elif self.output_layer in [3, 4, 5]:
                layer = self.output_layer - 3
                masks = masks[:, layer:]
                low_res_masks = low_res_masks[:, layer:]
                iou_preds = iou_preds[:, layer:]

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            # points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            low_res_masks=low_res_masks.flatten(0, 1)
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    def _process_sel_batch(
        self,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
    ) -> MaskData:

        if point_coords is None and boxes is None and mask_input is None:
            return MaskData()

        orig_h, orig_w = orig_size

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.predictor.transform.apply_coords(point_coords, im_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.predictor.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.predictor.device)
            if len(coords_torch.shape) == 2:
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if boxes is not None:
            box = self.predictor.transform.apply_boxes(boxes, im_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.predictor.device)
            if len(box_torch.shape) == 1:
                box_torch = box_torch[None, :]
            if box_torch.shape[0] == 1:
                box_torch = box_torch.expand(coords_torch.shape[0], -1)
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.predictor.device)
            if len(mask_input_torch.shape) == 3:
                mask_input_torch = mask_input_torch[None, :, :, :]
            if mask_input_torch.shape[0] == 1:
                mask_input_torch = mask_input_torch.expand(coords_torch.shape[0], -1, -1, -1)

        # Run model on this batch
        masks, iou_preds, low_res_masks, _ = self.predictor.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output=self.sel_multimask_output,
            return_logits=True,
        )

        if self.sel_multimask_output:
            if self.sel_output_layer in [0, 1, 2]:
                masks = masks[:, self.sel_output_layer][:, None, :, :]
                low_res_masks = low_res_masks[:, self.sel_output_layer][:, None, :, :]
                iou_preds = iou_preds[:, self.sel_output_layer][:, None]

            elif self.sel_output_layer in [3, 4, 5]:
                layer = self.sel_output_layer - 3
                masks = masks[:, layer:]
                low_res_masks = low_res_masks[:, layer:]
                iou_preds = iou_preds[:, layer:]

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            low_res_masks=low_res_masks.flatten(0, 1)
            # points=torch.as_tensor(point_coords.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.sel_pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.sel_pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.sel_stability_score_offset
        )
        if self.sel_stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.sel_stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
