import os
import glob
import numpy as np
import sys
import carb
import argparse
from omni.isaac.kit import SimulationApp

# Default rendering parameters
RENDER_CONFIG = {"renderer": "PathTracing", "samples_per_pixel_per_frame": 12, "headless": False}
kit = SimulationApp(RENDER_CONFIG)

import omni.replicator.core as rep
from omni.isaac.shapenet import utils
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import set_stage_up_axis


def setup_world():

    """Setup the Isaac sim world: lights, walls, floor, ceiling
    """

    create_prim("/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]})
    create_prim(
        "/World/Ground",
        "Cylinder",
        position=np.array([0.0, -0.5, 0.0]),
        orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
        attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(0.2, 0.2, 0.2)]},
    )
    create_prim("/World/Asset", "Xform")

    # set stage up axis to Y-up
    set_stage_up_axis("y")

    kit.update()


def find_usd_assets(root, category):

    """Look for USD files under root/category for each category specified.
    """

    MSW_model_paths = []

    all_assets = glob.glob(os.path.join(root, category, "*/*.usd"), recursive=True)
    print(os.path.join(root, category, "*/*.usd"))
    # Filter out large files (which can prevent OOM errors during training)

    num_assets = len(all_assets)
    if num_assets == 0:
        raise ValueError(f"No USDs found for category {category}.")

    MSW_model_paths = all_assets
  
    return MSW_model_paths


def batch_visualize_MSW(root, category):

    setup_world()

    # convert ShapeNet categories to synset ID
    category_id = utils.LABEL_TO_SYNSET.get(category)

    if isinstance(category_id, str):
        print("category_id is str")

    MSW_model_paths = find_usd_assets(root, category_id)

    # try:
    #     MSW_model_paths = find_usd_assets(root, category_id)
    # except ValueError as err:
    #     carb.log_error(str(err))
    #     kit.close()
    #     sys.exit()

    # batch load and visualize MSW models
    with rep.randomizer.instantiate(MSW_model_paths, size=rep.distribution.uniform(100, 150), mode="scene_instance"):
        rep.modify.semantics([("class", category)])
        rep.modify.pose(
            position=rep.distribution.uniform((-100, 5, -100), (100, 5, 100)),
            # rotation=self.rep.distribution.uniform((-90, -180, -90), (90, 180, 90)),
            # scale=self.rep.distribution.uniform(30, 50),
        )
        # self.rep.randomizer.texture(self._get_textures(), project_uvw=True)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser("MSW Models Visualization")
    # parser.add_argument("--category", type=str, required=True, help="List of object classes to use")
    # parser.add_argument("--root", type=str, default=None, help="Root directory containing USDs.")
    
    # args, unknown_args = parser.parse_known_args()

    # if args.root is None:
    #     if "WASTENET_LOCAL_DIR" in os.environ:
    #         wastenet_local_dir = f"{os.path.abspath(os.environ['WASTENET_LOCAL_DIR'])}_mat"
    #         if os.path.exists(wastenet_local_dir):
    #             args.root = wastenet_local_dir
    #     if args.root is None:
    #         print(
    #             "root argument not specified and WASTENET_LOCAL_DIR environment variable was not set or the path did not exist"
    #         )
    #         exit()
    
    # batch_visualize_MSW(args.root, args.category)

    root = '/media/walker2/ZHUOLI/DPS/dataset/WasteNet_nomat'
    category = 'can'

    batch_visualize_MSW(root, category)

    if True:
        kit.update

    



