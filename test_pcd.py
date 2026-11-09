from pynodes import Runtime
from pyslam.nodes.pcd_display_node import PointcloudNode, PointcloudGeneratorNode


if __name__ == '__main__':
    runtime = Runtime()
    runtime.add(PointcloudGeneratorNode)
    runtime.add(PointcloudNode)
    runtime.run()
