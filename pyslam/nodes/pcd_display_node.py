import time
import queue
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
        self.cam_poses = [np.eye(4)]
        self.cam_poses_ids = set()

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
        # Direct conversion is faster if numpy arrays are contiguous float64
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
            # Update existing geometry in-place instead of O(N^2) concatenation
            target_pcd = self.pcds_dict[pcd_id]
            target_pcd.points = received_pcd.points
            if received_pcd.has_colors():
                target_pcd.colors = received_pcd.colors
            self.viz.update_geometry(target_pcd)
        else:
            self.logger.info("Adding geometry")
            self.pcds_dict[pcd_id] = received_pcd
            self.viz.add_geometry(received_pcd)

        self.logger.info("Updating window")
        
        # Only update the view to the most recent pose once per PCD update
        for pose in self.cam_poses:
            self.update_view(pose)
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.002)

    def on_cam_pose(self, msg: Message) -> None:
        if msg.payload.get('done', False):
            return
            
        kfids = msg.payload['kf_ids']
        poses = msg.payload['pose']
        
        self.cam_poses = []
        for i, kid in enumerate(kfids):
            if kid not in self.cam_poses_ids:
                self.cam_poses_ids.add(kid)
                self.cam_poses.append(poses[i])

    def update_view(self, in_pose: np.ndarray) -> None:
        pose = np.eye(4)
        pose[:3] = in_pose[:3]

        view_ctrl = self.viz.get_view_control()
        cam = view_ctrl.convert_to_pinhole_camera_parameters()
        cam.extrinsic = pose

        view_ctrl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

    def spin(self):
        self.setup()
        self.logger.info("Started")

        while not self.stop_event.is_set() and not getattr(self, 'done', False):
            try:
                event = self.inbox.get(timeout=0.2)
                self._dispatch(event)
            except queue.Empty:
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
