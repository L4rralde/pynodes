from pyslam.utilities.shmem import AllocatedSharedMemoryNdarray


class AllocatedShmImage(AllocatedSharedMemoryNdarray):
    @staticmethod
    def attach(metadata):
        b, img = AllocatedSharedMemoryNdarray.attach(metadata)
        img_cp = img.copy()
        b.close()

        return img_cp
    