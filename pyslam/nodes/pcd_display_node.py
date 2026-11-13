import time
import queue
from typing import Dict, List

import open3d as o3d
import numpy as np

from pynodes import Node, Message
from pyslam.utilities.shmem import AllocatedShmPcdData


class PointcloudNode(Node):
    name = "pointcloud"

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.viz = None
        self.pcds_dict: Dict[str, object]= {}
        self.cam_poses: List[np.ndarray] = [np.eye(4)]

    def setup(self):
        self.subscribe("/cam/pose", self.on_cam_pose)
        self.subscribe("/pointcloud", self.on_pcd)
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=1280, height=720)
        self.setup_viz()

    def setup_viz(self) -> None:
        null_pcd = o3d.geometry.PointCloud()
        null_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3))) # Vacía
        self.viz.add_geometry(null_pcd)
        render_option = self.viz.get_render_option()
        render_option.background_color = np.array([0.0, 0.0, 0.0])

    def extract_msg_pcd_data(self, msg: Message) -> Dict[str, np.ndarray]:
        data = msg.payload
        shm = data.get('shm', False)
        if not shm:
            return data['pcd_data']

        buff, pcd = AllocatedShmPcdData.attach(data['pcd_data'])
        return buff, pcd

    def convert_msg_to_o3d_pcd(self, msg: Message) -> object:
        self.logger.info(f"Receiving new pointcloud: {msg.payload['pcd_id']}")
        buffer, msg.payload['pcd_data'] = self.extract_msg_pcd_data(msg)
        self.logger.info("Converting pcd data into open3d pcd")
        points = msg.payload['pcd_data']['points']
        colors = msg.payload['pcd_data']['colors']
        received_pcd = o3d.geometry.PointCloud()
        received_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            received_pcd.colors = o3d.utility.Vector3dVector(colors)
        if buffer is not None:
            buffer.close()
        return received_pcd

    def on_pcd(self, msg: Message) -> None:
        data = msg.payload
        done = data.get('done', False)
        if done:
            self.finish()
            return

        pcd_id = data['pcd_id']
        received_pcd = self.convert_msg_to_o3d_pcd(msg)

        if pcd_id in self.pcds_dict:
            full_pcd = None
            for pcd in self.pcds_dict.values():
                if full_pcd is None:
                    full_pcd = pcd
                    continue
                full_pcd += pcd
            self.viz.update_geometry(full_pcd)
        else:
            self.logger.info("Adding geometry")
            self.viz.add_geometry(received_pcd)

        self.pcds_dict[pcd_id] = received_pcd
        self.logger.info("Updating window")
        #self.apply_yaxis_rotation()
        for pose in self.cam_poses:
            self.update_view(pose)
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.001)

    def on_cam_pose(self, msg: Message) -> None:
        done = msg.payload.get('done', False)
        if done:
            return
        self.cam_poses = msg.payload['pose'] # n4 times 4 matrix

    def update_view(self, in_pose: np.ndarray) -> None:
        pose = np.eye(4)
        pose[:3] = in_pose
        # Open3D expects world-to-camera extrinsic
        extrinsic = pose

        view_ctrl = self.viz.get_view_control()
        cam = view_ctrl.convert_to_pinhole_camera_parameters()

        cam.extrinsic = extrinsic

        view_ctrl.convert_from_pinhole_camera_parameters(
            cam,
            allow_arbitrary=True
        )

    def spin(self):
        self.setup()
        self.logger.info("Started")

        while not self.stop_event.is_set() and not self.done:
            try:
                event = self.inbox.get(timeout=0.2)
                self._dispatch(event)
            except queue.Empty:
                self.viz.poll_events()
                continue

        self.logger.info("Stopped")


    def finish(self) -> None:
        self.viz.destroy_window()
        self.logger.info("Finishing")
        self.done = True


class PointcloudGeneratorNode(Node):
    name = "pointcloud_generator"

    def setup(self):
        self.advertise("/pointcloud")

    def finish(self) -> None:
        self.logger.info("Finishing")
        self.done = True
        self.publish(
            "/pointcloud",
            {
                "done": True
            }
        )

    def spin(self):
        self.setup()

        dataset = o3d.data.EaglePointCloud()
        pcd = o3d.io.read_point_cloud(dataset.path)
        original_points = np.asarray(pcd.points).copy()
        start_time = time.time()
        while not self.stop_event.is_set():
            elapsed = time.time() - start_time
        
            # Calculate a simple oscillating translation effect
            offset = np.sin(elapsed * 3.0) * 0.2

            updated_points = original_points.copy()
            updated_points[:, 2] += offset  # Animate along the Z-axis

            # Assign back to the point cloud geometry
            #pcd.points = o3d.utility.Vector3dVector(updated_points)
            self.publish(
                "/pointcloud",
                {
                    "points": updated_points,
                    "colors": np.random.uniform(0, 1, updated_points.shape)
                }
            )

            time.sleep(1)

        self.finish()
