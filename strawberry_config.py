from mrcnn.config import Config

class StrawberryConfig(Config):
    """
    用于训练玩具形状数据集的配置。
    从基本的Config类派生，并重写特定于玩具形状数据集的值。
    """
    # Give the configuration a recognizable name
    NAME = "strawberry"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768
    MAX_GT_INSTANCES = 100
    RPN_ANCHOR_SCALES = (8 * 7, 16 * 7, 32 * 7, 64 * 7, 128 * 7)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 100
    POST_NMS_ROIS_INFERENCE = 250
    POST_NMS_ROIS_TRAINING = 500
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = 0.01
