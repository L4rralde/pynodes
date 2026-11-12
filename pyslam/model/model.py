from typing import List, Dict

import torch
import torchvision.transforms.v2 as T
import numpy as np

import pyslam.conf as slam_conf


class VeggietSaladOmega:
    def __init__(self):
        self.model: torch.nn.Module|None = None

    def load_model(self, vpr_repo_path: str, ckpt_path: str) -> None:
        self.model = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            'vggto_s_pre',
            vpr_repo_path,
            ckpt_path
        ).eval().to('cuda')

    def model_prediction(self, images: torch.Tensor, kfids: List[int]=[], extract_features: bool=True) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            predictions = self.model.forward(images)
        
        extrinsics, intrinsics = self.model.backbone.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
        predictions.pop("pose_enc")

        predictions['extrinsic'] = extrinsics
        predictions['intrinsic'] = intrinsics

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        if not extract_features:
            predictions.pop('descriptor', None) #Skip computation if not required

        predictions['ids'] = np.asarray(kfids)
        return predictions
        