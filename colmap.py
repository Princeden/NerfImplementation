import pycolmap


class COLMAPPipeline:
    """
    Initalize the pipeline
    image_dir: directory path for images
    output_dir: directory to create path
    """

    def __init__(self, image_dir, output_dir):
        self.image_dir = image_dir
        self.output_dir = output_dir
