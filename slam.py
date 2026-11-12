from typing import Tuple, Dict, List, Set, Type
import time
from collections import defaultdict
import queue

import numpy as np
import cv2
import torch
from torchvision.transforms import v2 as T

from pynodes import Node, Runtime, Message
from pyslam import conf as slam_conf
from vggt_salad_slam.dtypes import Prediction
from vggt_salad_slam.transforms.estimate import vggtlong_est_scenes_transform
from vggt_salad_slam.transforms.graphs import Sim3Graph
from vggt_salad_slam.transforms import mattransforms as matt
from vggt_salad_slam.transforms.utils import to_pointcloud, to_ndarrays_for_pointcloud
from pyslam.nodes import (
    KeyFrameNode,
    ShowVideo,
    VprNode,
    LoopDetectionNode,
    FoundationModelNode,
    PointcloudNode
)

#TODO: Include as a thread in the FoundationModelNode
class LocalAlignNode(Node):
    name = "local_align"
    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.predictions_storage: Dict[int, Prediction] = {}
        self.ids_storage: Dict[int, List[int]]
        self._aligner_type: Type[matt.MatrixTransform] = matt.Sim3
        self.accumulated_transforms: Dict[int, matt.MatrixTransform] = {
            0: self._aligner_type()
        }

    def setup(self) -> None:
        self.advertise('/graph/measurement')
        self.advertise('/pointcloud')
        self.advertise('/cam/pose')
        self.subscribe(
            '/3dgfm/predictions',
            self.on_predictions
        )

    def append(self, msg: Message) -> None:
        data = msg.payload
        pred_type = data['type']
        pred_id = data['chunk_id']
        preds = data['predictions']

        if pred_type == 'loop_closure':
            raise RuntimeError("We don't store closed-loop predictions")

        if pred_id in self.predictions_storage.keys():
            raise RuntimeError(f"Predictions (id: {pred_id}) already in storage")

        self.predictions_storage[pred_id] = preds

    def get_predictions(self, pred_id: int) -> Prediction:
        return self.predictions_storage[pred_id]

    def on_predictions(self, msg: Message) -> None:
        data = msg.payload
        done = data.get("done", False)
        if done:
            self.finish()
            return

        pred_type = data['type']
        if pred_type == 'measurement':
            self.on_adjacent_prediction(msg)
        elif pred_type == "loop_closure":
            self.on_closed_loop_prediction(msg)
        else:
            raise ValueError(f"Incorrect pred type: {pred_type}")

    def on_adjacent_prediction(self, msg: Message) -> None:
        if self.predictions_storage:
            child_id = msg.payload['chunk_id']
            parent_id = child_id - 1
            self.logger.info(f"Aligning {child_id} with {parent_id}")
            parent_preds = self.get_predictions(parent_id)
            child_preds = msg.payload['predictions']
            self.logger.info(f"Parent ids: {parent_preds.ids}")
            self.logger.info(f"Child ids: {child_preds.ids}")
            aligner = self.compute_align(parent_preds, child_preds)
            parent_acc_pose = self.accumulated_transforms[parent_id]
            child_acc_pose = parent_acc_pose @ aligner
            self.accumulated_transforms[child_id] = child_acc_pose

        self.append(msg)
        chunk_id = msg.payload['chunk_id']
        pose = self.accumulated_transforms[chunk_id]
        preds = self.predictions_storage[chunk_id]
        global_preds = pose(preds)
        pcd_data = to_ndarrays_for_pointcloud(global_preds, lower_p=60)
        first_cam_pose = global_preds.extrinsic[0]
        self.publish(
            '/cam/pose',
            {
                'pose': first_cam_pose
            }
        )
        
        self.publish(
            "/pointcloud",
            {
                'pcd_id': chunk_id,
                **pcd_data
            }
        )

    def on_closed_loop_prediction(self, msg: Message) -> None:
        data = msg.payload

        connecting_preds = data['predictions']
        parent_id = data['edge_info']['parent']
        parent_preds = self.get_predictions(parent_id)
        child_id = data['edge_info']['child']
        child_preds = self.get_predictions(child_preds)

        aligner = (
            self.compute_align(parent_preds, connecting_preds) @
            self.compute_align(connecting_preds, child_preds)
        )

        self.publish(
            '/graph/measurement',
            {
                'type': 'loop_closure',
                'parent_id': parent_id,
                'child_id': child_id,
                'meas': aligner
            }
        )

    def finish(self) -> None:
        self.done = True
        self.logger.info("Finishing")
        self.publish(
            '/graph/measurement',
            {
                "done": True
            }
        )
        self.publish(
            '/pointcloud',
            {
                'done': True
            }
        )

    def compute_align(
        self,
        parent_preds: Prediction,
        child_preds: Prediction
    ) -> matt.MatrixTransform:

        aligner = vggtlong_est_scenes_transform(
            child_preds,
            parent_preds
        )
        return aligner


class PoseGraphNode(Node):
    name = "graph"
    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)

        self.graph = Sim3Graph()
        self.graph.add_anchor_prior(0)

        self.requires_optimization = False

    def setup(self) -> None:
        self.subscribe(
            '/graph/measurement',
            self.on_measurement
        )

    def on_measurement(self, msg: Message) -> None:
        data = msg.payload

        meas_type = data['type']

        self.graph.add_measurement(
            data['parent_id'],
            data['child_id'],
            data['meas']
        )

        if meas_type == 'loop_closure':
            self.requires_optimization = True

    def optimize_if_needed(self) -> None:
        if self.requires_optimization:
            _, new_est = self.graph.optimize(verbose=True)
            self.graph.update_estimation(new_est)
            self.requires_optimization = False

    def spin(self):
        self.setup()

        self.logger.info("Started")

        while not self.stop_event.is_set():
            try:
                event = self.inbox.get(timeout=0.2)
                self._dispatch(event)

            except queue.Empty:
                self.optimize_if_needed()

        self.logger.info("Stopped")


if __name__ == '__main__':
    runtime = Runtime()
    runtime.add(KeyFrameNode)
    runtime.add(ShowVideo)
    #runtime.add(VprNode)
    runtime.add(FoundationModelNode)
    runtime.add(PointcloudNode)
    runtime.add(LocalAlignNode)
    #runtime.add(LoopDetectionNode)
    runtime.run()
