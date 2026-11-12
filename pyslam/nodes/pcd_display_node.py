import time
import queue
from typing import Dict
import threading

import open3d as o3d
import numpy as np

from pynodes import Node, Message
from pyslam.utilities.shmem import AllocatedShmPcdData


class PointcloudNode(Node):
    name = "pointcloud"

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)

        self.pcd_queue: queue.Queue = queue.Queue()
        self.camera_lock: threading.Lock = threading.Lock()
        
        self.pcds_dict: Dict[str, object] = {}
        self.cam_poses = []
        
        self.viz = None
        self.viz_thread = None

    def setup(self):
        self.subscribe("/cam/pose", self.on_cam_pose)
        self.subscribe("/pointcloud", self.on_pcd)
        
        # Iniciar el hilo dedicado a la visualización
        self.viz_thread = threading.Thread(target=self.viz_loop, daemon=True)
        self.viz_thread.start()

    def viz_loop(self):
        """Ciclo principal de visualización que corre en su propio thread."""
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=1280, height=720)
        self.setup_viz()

        pose_idx = 0
        current_poses = []

        while not self.stop_event.is_set():
            # 1. Extraer pointclouds de la cola y actualizar geometría
            try:
                # Procesar todos los pcds pendientes en este frame
                while True:
                    pcd_id, received_pcd = self.pcd_queue.get_nowait()
                    
                    if pcd_id in self.pcds_dict:
                        self.pcds_dict[pcd_id] = received_pcd
                        # Combinar pcds
                        full_pcd = o3d.geometry.PointCloud()
                        for pcd in self.pcds_dict.values():
                            full_pcd += pcd
                        
                        # Open3D requiere actualizar la geometría explícitamente. 
                        # Nota: update_geometry asume que el número de puntos no cambió. 
                        # Si cambia, es mejor hacer clear_geometries() y add_geometry() o usar un contenedor fijo.
                        self.viz.clear_geometries()
                        self.viz.add_geometry(full_pcd)
                    else:
                        self.logger.info("Adding geometry")
                        self.viz.add_geometry(received_pcd)
                        self.pcds_dict[pcd_id] = received_pcd
                        
            except queue.Empty:
                pass

            # 2. Actualizar poses de la cámara de forma segura
            with self.camera_lock:
                if len(self.cam_poses) > 0:
                    current_poses = self.cam_poses.copy()
                    self.cam_poses.clear() # Limpiar para evitar reprocesar
                    pose_idx = 0

            # 3. Animar la cámara un paso por frame
            if current_poses and pose_idx < len(current_poses):
                self.update_view(current_poses[pose_idx])
                pose_idx += 1
                time.sleep(0.010) # Controlar la velocidad de la animación

            # 4. Renderizar
            self.viz.poll_events()
            self.viz.update_renderer()

        self.viz.destroy_window()

    def setup_viz(self) -> None:
        null_pcd = o3d.geometry.PointCloud()
        null_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.viz.add_geometry(null_pcd)
        render_option = self.viz.get_render_option()
        render_option.background_color = np.array([0.0, 0.0, 0.0])

    def extract_msg_pcd_data(self, msg) -> Dict[str, np.ndarray]:
        data = msg.payload
        shm = data.get('shm', False)
        if not shm:
            return None, data['pcd_data']

        buff, pcd = AllocatedShmPcdData.attach(data['pcd_data'])
        return buff, pcd

    def convert_msg_to_o3d_pcd(self, msg) -> object:
        self.logger.info(f"Receiving new pointcloud: {msg.payload['pcd_id']}")
        buffer, pcd_data = self.extract_msg_pcd_data(msg)
        
        self.logger.info("Converting pcd data into open3d pcd")
        points = pcd_data['points']
        colors = pcd_data.get('colors')
        
        received_pcd = o3d.geometry.PointCloud()
        received_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            received_pcd.colors = o3d.utility.Vector3dVector(colors)
        if buffer is not None:
            buffer.close()
        return received_pcd

    def on_pcd(self, msg) -> None:
        """Callback liberado: solo convierte y encola, no bloquea."""
        data = msg.payload
        if data.get('done', False):
            self.finish()
            return

        pcd_id = data['pcd_id']
        received_pcd = self.convert_msg_to_o3d_pcd(msg)
        
        # Enviar al hilo de visualización
        self.pcd_queue.put((pcd_id, received_pcd))

    def on_cam_pose(self, msg) -> None:
        if msg.payload.get('done', False):
            return
        
        with self.camera_lock:
            self.cam_poses = msg.payload['pose']

    def update_view(self, in_pose: np.ndarray) -> None:
        pose = np.eye(4)
        pose[:3] = in_pose

        view_ctrl = self.viz.get_view_control()
        cam = view_ctrl.convert_to_pinhole_camera_parameters()
        cam.extrinsic = pose
        view_ctrl.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)    

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
