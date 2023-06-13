import os
import glob
import torch
import numpy as np
import signal
import sys
import omni
import time

import carb

from omni.isaac.kit import SimulationApp

# Setup default variables
RESOLUTION = (1024, 1024)
# RESOLUTION = (1280, 720)
OBJ_LOC_MIN = (-50, 5, -50)
OBJ_LOC_MAX = (50, 5, 50)
CAM_LOC_MIN = (100, 0, -100)
CAM_LOC_MAX = (100, 100, 100)
SCALE_MIN = 15
SCALE_MAX = 40

# Default rendering parameters
RENDER_CONFIG = {"renderer": "PathTracing", "samples_per_pixel_per_frame": 12, "headless": True}


class RandomObjects(torch.utils.data.IterableDataset):
    """Dataset of random WasteNet objects.
    Objects are randomly chosen from selected categories and are positioned, rotated and coloured
    randomly in an empty room. RGB, BoundingBox2DTight and Instance Segmentation are captured by moving a
    camera aimed at the centre of the scene which is positioned at random at a fixed distance from the centre.

    Args:
        categories (tuple of str): Tuple or list of categories. For WasteNet, these will be the synset IDs.
        max_asset_size (int): Maximum asset file size that will be loaded. This prevents out of memory errors
            due to loading large meshes.
        num_assets_min (int): Minimum number of assets populated in the scene.
        num_assets_max (int): Maximum number of assets populated in the scene.
        split (float): Fraction of the USDs found to use for training.
        train (bool): If true, use the first training split and generate infinite random scenes.
    """

    def __init__(
        self, root, categories, max_asset_size=None, num_assets_min=3, num_assets_max=5, split=0.7, train=True
    ):
        assert len(categories) > 1
        assert (split > 0) and (split <= 1.0)

        self.kit = SimulationApp(RENDER_CONFIG)
        from omni.isaac.shapenet import utils
        import omni.replicator.core as rep
        import warp as wp

        self.rep = rep
        self.wp = wp

        # Convert WasteNet categories to synset ID
        category_ids = [utils.LABEL_TO_SYNSET.get(c, c) for c in categories]
        self.categories = category_ids
        self.range_num_assets = (num_assets_min, max(num_assets_min, num_assets_max))
        try:
            self.references = self._find_usd_assets(root, category_ids, max_asset_size, split, train)
        except ValueError as err:
            carb.log_error(str(err))
            self.kit.close()
            sys.exit()

        # Setup the scene, lights, walls, camera, etc.
        self.setup_scene()

        # Setup replicator randomizer graph
        self.setup_replicator()

        self.cur_idx = 0
        self.exiting = False

        signal.signal(signal.SIGINT, self._handle_exit)

    def _get_textures(self, texture_file):
        fi = open(texture_file, 'r')
        texture_path = fi.readlines()
        texture = []

        for t in texture_path:
            t = t.replace('\n','')
            texture.append(t)

        return texture    

    def _handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.exiting = True

    def setup_scene(self):
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.stage import set_stage_up_axis

        """Setup lights, walls, floor, ceiling and camera"""
        # Set stage up axis to Y-up
        set_stage_up_axis("y")

        # In a practical setting, the room parameters should attempt to match those of the
        # target domain. Here, we instead opt for simplicity.
        create_prim("/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]})

        create_prim(
            "/World/Ground",
            "Cylinder",
            position=np.array([0.0, -18.0, 0.0]),
            orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
            attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(0.0, 0.20, 0.05)]},
        )

        #setup scatter_ground for replicator to scatter MSW model(the scale of ground set to 300 is proper )
        self.scatter_ground = self.rep.create.plane(position=np.array([0.0, -0.5, 0.0]), rotation=(0,45,0), scale=220, visible=False)

        create_prim("/World/Asset", "Xform")

        self.camera = self.rep.create.camera()
        self.render_product = self.rep.create.render_product(self.camera, RESOLUTION)

        # Setup annotators that will report groundtruth
        self.rgb = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.bbox_2d_tight = self.rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        self.instance_seg = self.rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        self.rgb.attach(self.render_product)
        self.bbox_2d_tight.attach(self.render_product)
        self.instance_seg.attach(self.render_product)

        self.kit.update()

    def _find_usd_assets(self, root, categories, max_asset_size, split, train=True):
        """Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files found and select
        assets up to split * len(num_assets) if `train=True`, otherwise select the
        remainder.
        """
        references = {}
        for category in categories:
            all_assets = glob.glob(os.path.join(root, category, "*/*.usd"), recursive=True)
            # print("all_assets:", all_assets)
            print(os.path.join(root, category, "*/*.usd"))
            # Filter out large files (which can prevent OOM errors during training)
            if max_asset_size is None:
                assets_filtered = all_assets
            else:
                assets_filtered = []
                for a in all_assets:
                    if os.stat(a).st_size > max_asset_size * 1e6:
                        print(f"{a} skipped as it exceeded the max size {max_asset_size} MB.")
                    else:
                        assets_filtered.append(a)

            num_assets = len(assets_filtered)
            if num_assets == 0:
                raise ValueError(f"No USDs found for category {category} under max size {max_asset_size} MB.")

            if train:
                references[category] = assets_filtered[: int(num_assets * split)]
            else:
                references[category] = assets_filtered[int(num_assets * split) :]
        return references

    def _instantiate_category(self, category, references):
        with self.rep.randomizer.instantiate(references, size=self.rep.distribution.uniform(1, 3), mode="scene_instance", with_replacements=False):

            self.rep.physics.collider(approximation_shape="convexHull")

            self.rep.randomizer.scatter_2d(self.scatter_ground)

            self.rep.modify.semantics([("class", category)])

            self.rep.modify.pose(
                position=self.rep.distribution.uniform((-100, 5, -100), (100, 5, 100)),
                rotation=self.rep.distribution.uniform((90, -180, 0), (90, 180, 0)),
                # rotation=self.rep.distribution.uniform((-90, -180, -90), (90, 180, 90)),
                # rotation=self.rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
                scale=self.rep.distribution.uniform(75, 80),
                # scale=self.rep.distribution.uniform(45, 46),
            ) 

            self.rep.randomizer.texture(self._get_textures("/home/walker2/.local/share/ov/pkg/isaac_sim-2022.2.0/dps_ws/assests/texture/texture.txt"), project_uvw=True)

            # apply proper materials for the speciic category
            # if category == "bottle":
            # mats = self.rep.create.material_omnipbr(diffuse=self.rep.distribution.uniform((0.8, 0.8, 0.8, 0.2), (0.82, 0.82, 0.82)), roughness=self.rep.distribution.uniform(0.4, 0.41), metallic=0, count=100)
            # mats = self.rep.create.material_omnipbr(diffuse=self.rep.distribution.uniform((0,0,0), (1,1,1)), count=100)
            # self.rep.randomizer.materials(mats)

            # mats = self.rep.create.material_omnipbr(diffuse_texture=self.rep.distribution.choice(["omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.0/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
            #         "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.0/Isaac/Materials/Textures/Patterns/nv_brick_grey.jpg",
            #         "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.0/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
            #         "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.0/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
            #         "omniverse://localhost/NVIDIA/Assets/Isaac/2022.2.0/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg"]))
            # self.rep.randomizer.materials(mats)
            
            # if category == "can":
            #     mats = self.rep.create.material_omnipbr(diffuse=self.rep.distribution.uniform((0,0,0), (1,1,1)), count=100)
            #     self.rep.randomizer.materials(mats)

    def setup_replicator(self):
        """Setup the replicator graph with various attributes."""

        # Create two sphere lights
        light1 = self.rep.create.light(light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0)
        light2 = self.rep.create.light(light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0)

        with self.rep.new_layer():
            with self.rep.trigger.on_frame():

                # Randomize camera position
                with self.camera:
                    self.rep.modify.pose(
                        position=self.rep.distribution.uniform((1, 300, 1), (1, 300, 1)), look_at=(0, 0, 0)
                    )

                # Randomize asset positions and textures
                for category, references in self.references.items():
                    self._instantiate_category(category, references)

        # Run replicator for a single iteration without triggering any writes
        self.rep.orchestrator.preview()

    def setup_writer(self):

        # Initialize and attach writer
        writer = self.rep.WriterRegistry.get("BasicWriter")
        output_directory = os.getcwd() + "/_output_headless"
        print("Outputting data to ", output_directory)
        writer.initialize(
            output_dir=output_directory,
            rgb=True,
            bounding_box_2d_tight=True,
            instance_segmentation=True,
        )

        writer.attach([self.render_product])

        # run_orchestrator()
        # simulation_app.update()

    
    def __iter__(self):
        return self

    def __next__(self):
        # Step - trigger a randomization and a render
        self.rep.orchestrator.step()

        # Collect Groundtruth
        gt = {
            "rgb": self.rgb.get_data(device="cuda"),
            "boundingBox2DTight": self.bbox_2d_tight.get_data(device="cpu"),
            "instanceSegmentation": self.instance_seg.get_data(device="cuda"),
        }

        # RGB
        # Drop alpha channel
        image = self.wp.to_torch(gt["rgb"])[..., :3]

        # Normalize between 0. and 1. and change order to channel-first.
        image = image.float() / 255.0
        image = image.permute(2, 0, 1)

        # Bounding Box
        gt_bbox = gt["boundingBox2DTight"]["data"]

        # Create mapping from categories to index
        bboxes = torch.tensor(gt_bbox[["x_min", "y_min", "x_max", "y_max"]].tolist(), device="cuda")
        id_to_labels = gt["boundingBox2DTight"]["info"]["idToLabels"]
        prim_paths = gt["boundingBox2DTight"]["info"]["primPaths"]

        # For each bounding box, map semantic label to label index
        cat_to_id = {cat: i + 1 for i, cat in enumerate(self.categories)}
        semantic_labels_mapping = {int(k): v.get("class", "") for k, v in id_to_labels.items()}
        semantic_labels = [cat_to_id[semantic_labels_mapping[i]] for i in gt_bbox["semanticId"]]
        labels = torch.tensor(semantic_labels, device="cuda")

        # Calculate bounding box area for each area
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # Identify invalid bounding boxes to filter final output
        valid_areas = (areas > 0.0) * (areas < (image.shape[1] * image.shape[2]))

        # Instance Segmentation
        instance_data = self.wp.to_torch(gt["instanceSegmentation"]["data"]).squeeze()
        path_to_instance_id = {v: int(k) for k, v in gt["instanceSegmentation"]["info"]["idToLabels"].items()}

        instance_list = [im[0] for im in gt_bbox]
        masks = torch.zeros((len(instance_list), *instance_data.shape), dtype=bool, device="cuda")

        # Filter for the mask of each object
        for i, prim_path in enumerate(prim_paths):
            # Merge child instances of prim_path as one instance
            for instance in path_to_instance_id:
                if prim_path in instance:
                    masks[i] += torch.isin(instance_data, path_to_instance_id[instance])

        target = {
            "boxes": bboxes[valid_areas],
            "labels": labels[valid_areas],
            "masks": masks[valid_areas],
            "image_id": torch.LongTensor([self.cur_idx]),
            "area": areas[valid_areas],
            "iscrowd": torch.BoolTensor([False] * len(bboxes[valid_areas])),  # Assume no crowds
        }

        self.cur_idx += 1

        # delete caches
        del bboxes, masks, labels
        torch.cuda.empty_cache()

        return image, target


if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument("--categories", type=str, nargs="+", required=True, help="List of object classes to use")
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument(
        "--num_test_images", type=int, default=20, help="number of test images to generate when executing main"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing USDs. If not specified, use {SHAPENET_LOCAL_DIR}_mat as root.",
    )
    args, unknown_args = parser.parse_known_args()

    dataset = RandomObjects(args.root, args.categories, max_asset_size=args.max_asset_size)
    from omni.isaac.synthetic_utils import visualization
    from omni.isaac.shapenet import utils

    categories = [utils.LABEL_TO_SYNSET.get(c, c) for c in args.categories]

    # Iterate through dataset and visualize the output
    plt.ion()
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()

    # Directory to save the example images to
    now = time.localtime()
    year = now.tm_year
    month = now.tm_mon
    day = now.tm_mday
    out_dir = os.path.join(os.getcwd(), "dps_ws/assests", "_out_gen_imgs", f"{year}.{month}.{day}", "")
    os.makedirs(out_dir, exist_ok=True)

    image_num = 0

    # dataset.setup_writer()

    for image, target in dataset:
        for ax in axes:
            ax.clear()
            ax.axis("off")

        np_image = image.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np_image)

        num_instances = len(target["boxes"])
        colours = visualization.random_colours(num_instances)
        overlay = np.zeros_like(np_image)
        for mask, colour in zip(target["masks"].cpu().numpy(), colours):
            overlay[mask, :3] = colour

        axes[1].imshow(overlay)
        mapping = {i + 1: cat for i, cat in enumerate(categories)}
        labels = [utils.SYNSET_TO_LABEL[mapping[label.item()]] for label in target["labels"]]
        visualization.plot_boxes(ax, target["boxes"].tolist(), labels=labels, colours=colours)

        plt.draw()
        plt.pause(0.01)
        fig_name = os.path.join(out_dir, f"domain_randomization_test_image_{image_num}.png")
        plt.savefig(fig_name)
        image_num += 1
        if dataset.exiting or (image_num >= args.num_test_images):
            break

    # cleanup
    dataset.kit.close()
