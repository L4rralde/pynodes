import time
import queue
from typing import Dict

import open3d as o3d
import numpy as np

from pynodes import Node, Message


class PointcloudNode(Node):
    name = "pointcloud"

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.viz = None
        self.pcds_dict: Dict[str, object]= {}
        self.cam_pose: np.ndarray = np.eye(4)

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
        
    def apply_yaxis_rotation(self) -> None:
        view_ctrl = self.viz.get_view_control()
        self.viz.reset_view_point()
        theta = np.radians(180)  # 30-degree rotation
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0,             1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        cam = view_ctrl.convert_to_pinhole_camera_parameters()

        extrinsic = cam.extrinsic.copy()
        extrinsic[:3, :3] = R_y
        cam.extrinsic = extrinsic

        view_ctrl.convert_from_pinhole_camera_parameters(cam)

    def on_pcd(self, msg: Message) -> None:
        data = msg.payload
        done = data.get('done', False)
        if done:
            self.finish()
            return

        pcd_id = data['pcd_id']
        points = data['points']
        colors = data.get('colors', None)
        self.logger.info(f"Receiving new pointcloud: {pcd_id}")

        received_pcd = o3d.geometry.PointCloud()
        received_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            received_pcd.colors = o3d.utility.Vector3dVector(colors)

        if pcd_id in self.pcds_dict:
            full_pcd = None
            for pcd in self.pcds_dict.values():
                if full_pcd is None:
                    full_pcd = pcd
                    continue
                full_pcd += pcd
            self.viz.update_geometry(full_pcd)
        else:
            self.viz.add_geometry(received_pcd)
            self.logger.info("Adding geometry")

        #self.apply_yaxis_rotation()
        self.update_view()
        self.pcds_dict[pcd_id] = received_pcd
        self.viz.poll_events()
        self.viz.update_renderer()

    def on_cam_pose(self, msg: Message) -> None:
        done = msg.payload.get('done', False)
        if done:
            return
        self.cam_pose[:3] = msg.payload['pose'] # 4 times 4 matrix

    def update_view(self) -> None:
        pose = self.cam_pose  # T_world_cam

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
