import time
import queue
from collections import deque
from typing import Dict

import open3d as o3d
import numpy as np

from pynodes import Node, Message
from pyslam.utilities.shmem import AllocatedShmPcdData


class PointcloudNode(Node):
    name = "pointcloud"

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.viz = None
        self.pcds_dict: Dict[str, o3d.geometry.PointCloud] = {}
        
        # Use a deque for efficient popping from the left
        self.cam_poses_queue = deque()
        self.cam_poses_ids = set()
        
        # --- Stabilization attributes ---
        self.smoothed_pose = None
        self.alpha = 0.4  # Lower alpha for smoother continuous fly-throughs

    def setup(self):
        self.subscribe("/cam/pose", self.on_cam_pose)
        self.subscribe("/pointcloud", self.on_pcd)
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=1280, height=720)
        self.setup_viz()

    def setup_viz(self) -> None:
        null_pcd = o3d.geometry.PointCloud()
        null_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.viz.add_geometry(null_pcd)
        render_option = self.viz.get_render_option()
        render_option.background_color = np.zeros(3)

    def extract_msg_pcd_data(self, msg: Message):
        data = msg.payload
        if not data.get('shm', False):
            return None, data['pcd_data']

        return AllocatedShmPcdData.attach(data['pcd_data'])

    def convert_msg_to_o3d_pcd(self, msg: Message) -> o3d.geometry.PointCloud:
        self.logger.info(f"Receiving new pointcloud: {msg.payload['pcd_id']}")
        buffer, pcd_data = self.extract_msg_pcd_data(msg)
        
        received_pcd = o3d.geometry.PointCloud()
        received_pcd.points = o3d.utility.Vector3dVector(pcd_data['points'])
        
        colors = pcd_data.get('colors')
        if colors is not None:
            received_pcd.colors = o3d.utility.Vector3dVector(colors)
            
        if buffer is not None:
            buffer.close()
            
        return received_pcd

    def on_pcd(self, msg: Message) -> None:
        data = msg.payload
        if data.get('done', False):
            self.finish()
            return

        pcd_id = data['pcd_id']
        received_pcd = self.convert_msg_to_o3d_pcd(msg)

        if pcd_id in self.pcds_dict:
            target_pcd = self.pcds_dict[pcd_id]
            target_pcd.points = received_pcd.points
            if received_pcd.has_colors():
                target_pcd.colors = received_pcd.colors
            self.viz.update_geometry(target_pcd)
        else:
            self.logger.info("Adding geometry")
            self.pcds_dict[pcd_id] = received_pcd
            self.viz.add_geometry(received_pcd)

        # Rendering is completely removed from this callback to prevent UI blocking

    def on_cam_pose(self, msg: Message) -> None:
        if msg.payload.get('done', False):
            return
            
        kfids = msg.payload['kf_ids']
        poses = msg.payload['pose']
        
        for i, kid in enumerate(kfids):
            if kid not in self.cam_poses_ids:
                self.cam_poses_ids.add(kid)
                if i % 2 == 0:
                    self.cam_poses_queue.append(poses[i])

    def update_view(self, in_pose: np.ndarray) -> None:
        if self.smoothed_pose is None:
            self.smoothed_pose = np.eye(4, dtype=np.float64)
            self.smoothed_pose[:3] = in_pose[:3]
        else:
            # 1. Fast Translation Lerp
            self.smoothed_pose[:3, 3] += self.alpha * (in_pose[:3, 3] - self.smoothed_pose[:3, 3])
            
            # 2. Fast Rotation Matrix interpolation via SVD
            R_current = self.smoothed_pose[:3, :3]
            R_target = in_pose[:3, :3]
            
            R_blended = (1.0 - self.alpha) * R_current + self.alpha * R_target
            
            # Orthogonalize to ensure it remains a valid rotation matrix
            U, _, Vt = np.linalg.svd(R_blended)
            R_valid = U @ Vt
            
            # Handle reflection case
            if np.linalg.det(R_valid) < 0:
                U[:, 2] *= -1
                R_valid = U @ Vt
                
            self.smoothed_pose[:3, :3] = R_valid

        # Update Open3D camera
        view_ctrl = self.viz.get_view_control()
        cam = view_ctrl.convert_to_pinhole_camera_parameters()
        cam.extrinsic = self.smoothed_pose.astype(np.float64)
        view_ctrl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    def spin(self):
        self.setup()
        self.logger.info("Started")

        while not self.stop_event.is_set() and not getattr(self, 'done', False):
            try:
                # Process incoming messages with a very fast timeout
                event = self.inbox.get(timeout=0.01)
                self._dispatch(event)
            except queue.Empty:
                pass
                
            # --- Trajectory Interpolation Logic ---
            if self.cam_poses_queue:
                target_pose = self.cam_poses_queue[0]
                self.update_view(target_pose)
                
                # Calculate positional distance to the current target
                dist = np.linalg.norm(self.smoothed_pose[:3, 3] - target_pose[:3, 3])
                
                # If close enough (e.g., 0.05 units), or queue is getting backed up, 
                # pop the target to chase the next waypoint.
                if dist < 0.05 or len(self.cam_poses_queue) > 10:
                    self.cam_poses_queue.popleft()
                    
            # Render on the main thread exactly once per loop
            self.viz.poll_events()
            self.viz.update_renderer()

        self.logger.info("Stopped")

    def finish(self) -> None:
        if self.viz is not None:
            while self.viz.poll_events():
                self.viz.update_renderer()
                time.sleep(0.01)
            self.viz.destroy_window()
        self.logger.info("Finishing")
        self.done = True
