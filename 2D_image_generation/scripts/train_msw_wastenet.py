"""Instance Segmentation Training of MSW Dataset for Waste Analysis

Use a PyTorch dataloader together with OmniKit to generate scenes and groundtruth to
train a [Mask-RCNN](https://arxiv.org/abs/1703.06870) model.
"""

import torch, gc
import os
import matplotlib.pyplot as plt
import numpy as np
import signal


from generate_msw_wastenet import RandomObjects


def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.cuda.set_device(2)
    device = "cpu"

    train_set = RandomObjects(
        args.root, args.categories, num_assets_min=3, num_assets_max=5, max_asset_size=args.max_asset_size
    )

    def handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        train_set.exiting = True

    signal.signal(signal.SIGINT, handle_exit)

    from omni.isaac.synthetic_utils import visualization
    from omni.isaac.shapenet import utils
    import torch
    from torch.utils.data import DataLoader
    import torchvision

    # Setup data
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=lambda x: tuple(zip(*x)))

    # Setup Model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=1 + len(args.categories))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.visualize:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Directory to save the train images to
    out_dir = os.path.join(os.getcwd(), "_out_train_imgs", "")
    os.makedirs(out_dir, exist_ok=True)

    for i, train_batch in enumerate(train_loader):

        #clear cuda catch before each epoch, which aims to solve the CUDA Out Of Memory Error 
        gc.collect()
        torch.cuda.empty_cache()

        if i > args.max_iters or train_set.exiting:
            print("Exiting ...")
            train_set.kit.close()
            break

        model.train()
        images, targets = train_batch
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #use cudnn.benchmark, which aims to solve RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        print(f"ITER {i} | {loss:.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(images[:1])

            if args.visualize:
                idx = 0
                score_thresh = 0.5
                mask_thresh = 0.5

                pred = predictions[idx]

                np_image = images[idx].permute(1, 2, 0).cpu().numpy()
                for ax in axes:
                    fig.suptitle(f"Iteration {i:05}", fontsize=14)
                    ax.cla()
                    ax.axis("off")
                    ax.imshow(np_image)
                axes[0].set_title("Input")
                axes[1].set_title("Input + Predictions")

                score_filter = [i for i in range(len(pred["scores"])) if pred["scores"][i] > score_thresh]
                num_instances = len(score_filter)
                colours = visualization.random_colours(num_instances, enable_random=False)

                overlay = np.zeros_like(np_image)
                for mask, colour in zip(pred["masks"], colours):
                    overlay[mask.squeeze().cpu().numpy() > mask_thresh, :3] = colour

                axes[1].imshow(overlay, alpha=0.5)
                # If WasteNet categories are specified with their names, convert to synset ID
                args.categories = [utils.LABEL_TO_SYNSET.get(c, c) for c in args.categories]
                mapping = {i + 1: cat for i, cat in enumerate(args.categories)}
                labels = [utils.SYNSET_TO_LABEL[mapping[label.item()]] for label in pred["labels"]]
                visualization.plot_boxes(axes[1], pred["boxes"].cpu().numpy(), labels=labels, colours=colours)

                plt.draw()
                fig_name = os.path.join(out_dir, f"train_image_{i}.png")
                plt.savefig(fig_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Dataset test")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing ShapeNet USDs. If not specified, use {SHAPENET_LOCAL_DIR}_nomat as root.",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", required=True, help="List of ShapeNet categories to use (space seperated)."
    )
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_iters", type=float, default=1000, help="Number of training iterations.")
    parser.add_argument("--visualize", action="store_true", help="Visualize predicted masks during training.")
    args, unknown_args = parser.parse_known_args()

    # If root is not specified use the environment variable SHAPENET_LOCAL_DIR with the _nomat suffix as root
    if args.root is None:
        if "SHAPENET_LOCAL_DIR" in os.environ:
            shapenet_local_dir = f"{os.path.abspath(os.environ['SHAPENET_LOCAL_DIR'])}_nomat"
            if os.path.exists(shapenet_local_dir):
                args.root = shapenet_local_dir
        if args.root is None:
            print(
                "root argument not specified and SHAPENET_LOCAL_DIR environment variable was not set or the path did not exist"
            )
            exit()

    main(args)
