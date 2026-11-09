import time

import open3d as o3d
import numpy as np

from pynodes import Node, Message


class PointcloudNode(Node):
    name = "pointcloud"

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self._geometry_added: bool = False
        self.viz = None
        self.pcd = None

    def setup(self):
        self.subscribe("/pointcloud", self.on_pcd)
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=1280, height=720)
        self.pcd = o3d.geometry.PointCloud()

    def on_pcd(self, msg: Message) -> None:
        data = msg.payload
        done = data.get('done', False)
        if done:
            self.finish()
            return

        points = data['points']
        colors = data['colors']
        
        
        self.pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if not self._geometry_added:
            self.viz.add_geometry(self.pcd)
            self.viz.reset_view_point()
            self._geometry_added = True
        else:
            self.viz.update_geometry(self.pcd)

        self.viz.poll_events()
        self.viz.update_renderer()

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
