import time
import asyncio
import json
import io

import omni.kit
import omni.usd
import omni.replicator.core as rep

from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch

class COCOWriter(Writer):
    def __init__(
        self,
        output_dir,
        rgb: bool = True,
        bounding_box_2d_tight: bool = False,
        image_output_format="png",
    ):
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._image_output_format = image_output_format

        self.annotators = []

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight",
                                                                init_params={"semanticTypes": ["class"]}))

    def check_bbox_area(self, bbox_data, size_limit):
        length = abs(bbox_data['x_min'] - bbox_data['x_max'])
        width = abs(bbox_data['y_min'] - bbox_data['y_max'])

        area = length * width
        if area > size_limit:
            return True
        else:
            return False

    def write(self, data):
        if "rgb" in data and "bounding_box_2d_tight" in data:
            bbox_data = data["bounding_box_2d_tight"]["data"]
            id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]

            for id, labels in id_to_labels.items():
                id = int(id)

                if 'worker' in labels:

                    target_bbox_data = {'x_min': bbox_data['x_min'], 'y_min': bbox_data['y_min'],
                                        'x_max': bbox_data['x_max'], 'y_max': bbox_data['y_max']}

                    if self.check_bbox_area(target_bbox_data, 0.5):
                        width = int(abs(target_bbox_data["x_max"][0] - target_bbox_data["x_min"][0]))
                        height = int(abs(target_bbox_data["y_max"][0] - target_bbox_data["y_min"][0]))

                        if width != 2147483647 and height != 2147483647:
                            filepath = f"rgb_{self._frame_id}.{self._image_output_format}"
                            self._backend.write_image(filepath, data["rgb"])

                            bbox_filepath = f"bbox_{self._frame_id}.json"

                            coco_bbox_data = {"x": int(target_bbox_data["x_max"][0]),
                                            "y": int(target_bbox_data["y_max"][0]),
                                            "width": width,
                                            "height": height}

                            buf = io.BytesIO()
                            buf.write(json.dumps(coco_bbox_data).encode())
                            self._backend.write_blob(bbox_filepath, buf.getvalue())

        self._frame_id += 1