import cv2

from pynodes import Node, Message
from pyslam.utilities.shmem import AllocatedSharedMemoryNdarray


class ShowVideo(Node):
    name = "video_display"
    def setup(self):
        self.subscribe("/video/chunks", self.on_image)

    def on_image(self, msg: Message):
        done = msg.payload.get('done', False)
        if done:
            self.done = True
            cv2.destroyAllWindows()
            return
        #self.logger.info(f"Received kfids: {msg.payload['kf_ids']}")
        shm = msg.payload.get("shm", False)

        buffer = None
        if shm:
            buffer, imgs = AllocatedSharedMemoryNdarray.attach(msg.payload['imgs'])
        else:
            imgs = msg.payload['imgs']

        for img in imgs:
            cv2.imshow("frames", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if buffer is not None:
            buffer.close()

