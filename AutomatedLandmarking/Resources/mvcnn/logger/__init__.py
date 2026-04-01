from .logger import *

try:
    from .tensorboardutils import TensorboardWriter
except (ImportError, TypeError):
    # Stub when tensorboard not installed or protobuf incompatible (prediction-only use case)
    class TensorboardWriter:
        def __init__(self, log_dir, logger, enabled):
            self.writer = None  # No-op for prediction