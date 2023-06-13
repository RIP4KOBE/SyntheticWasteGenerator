"""Instance Segmentation Training of MSW Recognition Model[Mask-RCNN] for Waste Analysis
"""

import torch, gc
import os
import matplotlib.pyplot as plt
import numpy as np
import signal
import argparse
import torchvision
import time

from torch.utils.data import DataLoader
from generate_msw_wastenet import RandomObjects

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# get timestep for saving the trained model
now = time.localtime()
year = now.tm_year
month = now.tm_mon
day = now.tm_mday


def main(args):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # print("the current device is:", device)
    device = "cpu"

    # create training set
    train_set = RandomObjects(
        args.root, args.categories, num_assets_min=2, num_assets_max=5, max_asset_size=args.max_asset_size
    )

    def handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        train_set.exiting = True

    signal.signal(signal.SIGINT, handle_exit)

    # setup training data loader
    train_loader = DataLoader(train_set, batch_size=2, collate_fn=lambda x: tuple(zip(*x)))

    # setup MSW recognition Model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=1 + len(args.categories))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Directory to save the model weights
    model_dir = os.path.join(os.getcwd(), "dps_ws", "models", "")
    model_pth = os.path.join(model_dir, f"model_weights_{year}{month}{day}.pth")
    os.makedirs(model_dir, exist_ok=True)

    if args.visualize:
        plt.ion()

    # run the training loop
    for i, train_batch in enumerate(train_loader):

        # clear cuda catch before each epoch, which aims to solve the CUDA Out Of Memory Error 
        gc.collect()
        torch.cuda.empty_cache()

        if i > args.max_iters or train_set.exiting:
            torch.save(model.state_dict(), model_pth)
            print("Exiting ...")
            train_set.kit.close()
            break

        model.train()
        optimizer.zero_grad()
        images, targets = train_batch
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # use cudnn.benchmark, which aims to solve RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.allow_tf32 = True

        loss_dict = model(images, targets)
        # print("type of the loss_dict is:", type(loss_dict))
        loss = sum(loss for loss in loss_dict.values())
        # print("type of the loss is:", type(loss))
        print(f"ITER {i} | {loss:.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(images[:1])

            if args.visualize:
                visualize_predicted_results(predictions, i, images, args)
        
        # delete caches
        del images, targets, loss_dict, loss
        torch.cuda.empty_cache()


def visualize_predicted_results(predictions, iterator, original_images, args):
    from omni.isaac.synthetic_utils import visualization
    from omni.isaac.shapenet import utils

    # setup visualization parameters
    i = iterator
    images = original_images
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    idx = 0
    score_thresh = 0.5
    mask_thresh = 0.5
    pred = predictions[idx]
    np_image = images[idx].permute(1, 2, 0).cpu().numpy()

    # directory to save the visualized images to
    out_dir = os.path.join(os.getcwd(), "dps_ws/assests", "_out_train_imgs", f"{year}.{month}.{day}", "")
    os.makedirs(out_dir, exist_ok=True)

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

    #convert WasteNet categories to synset ID
    args.categories = [utils.LABEL_TO_SYNSET.get(c, c) for c in args.categories]
    mapping = {i + 1: cat for i, cat in enumerate(args.categories)}
    labels = [utils.SYNSET_TO_LABEL[mapping[label.item()]] for label in pred["labels"]]
    visualization.plot_boxes(axes[1], pred["boxes"].cpu().numpy(), labels=labels, colours=colours)

    plt.draw()
    fig_name = os.path.join(out_dir, f"train_image_{i}.png")
    plt.savefig(fig_name)

if __name__ == "__main__":
    #parse training-related args
    parser = argparse.ArgumentParser("MSW Recognition Model Training")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing WasteNet USDs.",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", required=True, help="List of WasteNet categories to use (space seperated)."
    )
    parser.add_argument(
        "--max_asset_size",
        type=float,
        default=10.0,
        help="Maximum asset size to use in MB. Larger assets will be skipped.",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_iters", type=float, default=500, help="Number of training iterations.")
    parser.add_argument("--visualize", action="store_true", help="Visualize predicted masks during training.")
    args, unknown_args = parser.parse_known_args()

    #start training
    main(args)
