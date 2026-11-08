from typing import List, Set, Dict
from collections import defaultdict

import numpy as np

from pynodes import Node, Message
import pyslam.conf as slam_conf


class LoopDetectionNode(Node):
    name = "loop_detection"
    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.descriptors_mat: np.ndarray|None = None
        self.kf_ids: List[int] = []
        self.kf_ids_set: Set[int] = set()
        self.chunk_dict: Dict[int, List[int]] = defaultdict(list)

    def setup(self):
        self.subscribe('/vpr/descriptors', self.on_descriptors)
        self.advertise('/closed_loop/closing_chunk')
        self.descriptors_mat = np.zeros((128, 8448))

    def get_descriptors(self) -> np.ndarray:
        return self.descriptors_mat[:len(self.kf_ids)]

    def append_chunk(self, msg: Message) -> None:
        chunk_id = msg.payload['chunk_id']
        kf_ids = msg.payload['kf_ids']
        descriptors = msg.payload['descriptors']

        if chunk_id != len(self.chunk_dict.keys()):
            raise RuntimeError(f"Incorrect chunk id: {chunk_id}")

        self.chunk_dict[chunk_id] = kf_ids

        new_descriptors = {
            kfid: desc
            for kfid, desc in zip(kf_ids, descriptors)
            if kfid not in self.kf_ids_set
        }
        self.logger.info(
            f"Appending new descriptors: chunk_id: {chunk_id}, {new_descriptors.keys()}"
        )

        for kfid, desc in new_descriptors.items():
            if len(self.kf_ids) + 1 > len(self.descriptors_mat):
                self.logger.warning("Reserving more memory")
                self.reserve_more_memory()                

            self.descriptors_mat[len(self.kf_ids)] = desc
            self.kf_ids.append(kfid)
            self.kf_ids_set.add(kfid)

    def reserve_more_memory(self) -> None:
        new_descriptors_mat = np.zeros(
            (2 * len(self.descriptors_mat), self.descriptors_mat.shape[1]),
            dtype=self.descriptors_mat.dtype
        )

        new_descriptors_mat[:len(self.kf_ids)] = (
            self.descriptors_mat[:len(self.kf_ids)]
        )

        self.descriptors_mat = new_descriptors_mat

    def similarity_search(self, msg: Message) -> None:
        ignore_kfs = (
            3 * (slam_conf.CHUNK_SIZE - slam_conf.CHUNK_OVERLAP)
            + slam_conf.CHUNK_OVERLAP
        )

        if ignore_kfs >= len(self.kf_ids):
            return ()

        descriptors = self.get_descriptors()

        descriptors = descriptors[:-ignore_kfs]

        incoming_descriptors = msg.payload['descriptors']

        sim = descriptors @ incoming_descriptors.T
        sim = np.nan_to_num(sim, nan=0.0)

        rows, cols = np.where(sim >= slam_conf.MIN_SIMILARITY)

        matches = [
            (
                self.kf_ids[row],
                msg.payload['kf_ids'][col],
                sim[row, col]
            )
            for row, col in zip(rows, cols)
        ]

        if not matches:
            return []

        matches = self.non_maximum_suppression(
            matches,
            slam_conf.NMS_WINDOW
        )

        self.logger.warning("Found matches")
        self.logger.warning(matches)

        return matches
    

    def non_maximum_suppression(self, matches, window):
        if not matches:
            return []

        # Convert to numpy array for fast vectorization
        matches_np = np.array(matches)
        
        # Get the indices that would sort the array by score (column 2) descending
        order = matches_np[:, 2].argsort()[::-1]
        
        selected = []
        
        while order.size > 0:
            # 1. Take the index of the highest scoring match
            i = order[0]
            
            # 2. Append it to our results (cast back to native Python types)
            kf_a = int(matches_np[i, 0])
            kf_b = int(matches_np[i, 1])
            score = float(matches_np[i, 2])
            selected.append((kf_a, kf_b, score))
            
            if order.size == 1:
                break
                
            # 3. Get the keyframes of all OTHER remaining matches
            remaining_a = matches_np[order[1:], 0]
            remaining_b = matches_np[order[1:], 1]
            
            # 4. Find which remaining matches are strictly OUTSIDE the window
            # We use bitwise OR (|) because a match is valid if it exceeds the 
            # window in AT LEAST ONE dimension.
            not_suppressed = (
                (np.abs(remaining_a - kf_a) > window) | 
                (np.abs(remaining_b - kf_b) > window)
            )
            
            # 5. Keep only the unsuppressed indices
            # (+1 is required because `remaining` arrays are offset by 1 from `order`)
            order = order[np.where(not_suppressed)[0] + 1]
            
        return selected

    def find_first_containing_chunk(self, kfid: int) -> int:
        for chunk_id, kfdis in self.chunk_dict.items():
            if kfid in kfdis:
                return chunk_id
        raise RuntimeError(f"Couldn't find a registered parent for kf id: {kfid}")

    def on_descriptors(self, msg: Message) -> None:
        done = msg.payload.get('done', False)
        if done:
            self.done = True
            self.logger.info("Finishing")
            return
        loop_pairs = self.similarity_search(msg)
        self.append_chunk(msg)
        for pair in loop_pairs:
            self.publish(
                '/closed_loop/closing_chunk',
                {
                    'kf_ids': [pair[0], pair[1]],
                    'sim': pair[2],
                    'child': self.find_first_containing_chunk(pair[0]),
                    'parent': msg.payload['chunk_id']
                }
            )
