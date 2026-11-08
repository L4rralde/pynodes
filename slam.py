from typing import Tuple, Dict, List, Set
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
from pyslam.nodes import (
    KeyFrameNode,
    ShowVideo,
    VprNode,
    LoopDetectionNode
)

#TODO: Include as a thread in the FoundationModelNode
class LocalAlignNode(Node):
    name = "local_align"
    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.predictions_storage: Dict[int, Dict[str, np.ndarray]] = []
        self.ids_storage: Dict[int, List[int]]

    def setup(self):
        self.advertise('/graph/measurement')
        self.subscribe(
            '/3dgfm/predictions',
            self.on_predictions
        )

    def append(self, msg: Message):
        data = msg.payload
        pred_type = data['type']
        pred_id = data['chunk_id']
        preds = data['predictions']

        if pred_type == 'loop_closure':
            raise RuntimeError("We don't store closed-loop predictions")

        if pred_id in self.predictions_storage.keys():
            raise RuntimeError(f"Predictions (id: {pred_id}) already in storage")

        self.predictions_storage[pred_id] = preds

    def get_predictions(self, pred_id: int) -> Dict[str, np.ndarray]:
        return self.predictions_storage[pred_id]

    def on_predictions(self, msg: Message):
        data = msg.payload
        pred_type = data['type']
        if pred_type == 'measurement':
            self.on_adjacent_prediction(msg)
            self.append(msg)
        elif pred_type == "loop_closure":
            self.on_closed_loop_prediction(msg)
        else:
            raise ValueError(f"Incorrect pred type: {pred_type}")

    def on_adjacent_prediction(self, msg: Message):
        child_id = msg.payload['chunk_id']
        parent_id = child_id - 1
        parent_preds = self.get_predictions(parent_id)
        child_preds = msg.payload['predictions']
        aligner = self.compute_align(parent_preds, child_preds)
        self.publish(
            '/graph/measurement',
            {
                'type': 'adjacent',
                'parent_id': parent_id,
                'child_id': child_id,
                'meas': aligner
            }
        )

    def on_closed_loop_prediction(self, msg: Message):
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

    def compute_align(
        self,
        parent_preds: Dict[str, np.ndarray],
        child_preds: Dict[str, np.ndarray]
    ) -> object:
        parent_preds = Prediction.from_dict(parent_preds)
        child_preds = Prediction.from_dict(child_preds)

        aligner = vggtlong_est_scenes_transform(
            parent_preds,
            child_preds
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
    runtime.add(VprNode)
    runtime.add(LoopDetectionNode)
    runtime.run()
