__all__ = ["config"]


class Config:
    def __init__(self):
        try:
            self.enable_gpu()
        except:
            self.gpu_enabled = False

    def enable_gpu(self):
        import cupy

        self.gpu_enabled = True

    def disable_gpu(self):
        self.gpu_enabled = False


config = Config()