import time

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pynodes import Node, Runtime


class CameraNode(Node):
    name="camera"

    def setup(self):
        self.advertise("/camera/image")

    def spin(self):
        self.setup()

        count = 0

        while not self.stop_event.is_set():
            self.publish(
                "/camera/image",
                {
                    "frame": count,
                    "pixels": [1, 2, 3, 4],
                },
            )

            print(f"[Camera] Published frame {count}", flush=True)

            count += 1
            time.sleep(0.05)


class ProcessingNode(Node):
    name = "processor"

    def setup(self):
        self.subscribe("/camera/image", self.on_image)

    def on_image(self, msg):
        print(
            f"[Processor] Received frame "
            f"{msg.header.sequence}: {msg.payload}",
            flush=True,
        )

    
if __name__ == '__main__':
    runtime = Runtime()
    runtime.add(CameraNode)
    runtime.add(ProcessingNode)

    runtime.run()
