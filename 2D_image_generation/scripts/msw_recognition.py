"""
MSW Recognition with Trained Mask-RCNN Model
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
import time

from omni.isaac.kit import SimulationApp

from PIL import Image

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"

# Default rendering parameters
RENDER_CONFIG = {"renderer": "PathTracing", "samples_per_pixel_per_frame": 12, "headless": True}

kit = SimulationApp(RENDER_CONFIG)


def main(args):
    # load real-world msw data
    msw_img_dir = args.root
    msw_img_list = load_msw_img(msw_img_dir)

    # load the trained msw recognition model
    model_pth = os.path.join(os.getcwd(), "dps_ws", "models", "model_weights_202363.pth")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=1 + len(args.categories))
    model = model.to(device)
    model.load_state_dict(torch.load(model_pth))
    model.eval()

    # initialize counter
    counter = 0

    # initialize visualizer
    if args.visualize:
        plt.ion()

    # run the recognition loop
    for img in msw_img_list:
        prediction = recognize(img, model, device)
        counter = counter + 1
        if args.visualize:
            visualize_recognition_results(prediction, counter, img, args)
        

def load_msw_img(msw_img_dir):
    msw_img_list = []
    msw_img_id = os.listdir(msw_img_dir)
    # print("the path of loaded msw img is:", msw_img_id[0])
    loader = transforms.Compose([transforms.ToTensor()]) 

    for msw_img in msw_img_id:

        # load the real-world msw images
        msw_img_pth = msw_img_dir + "/" + msw_img
        msw_img = Image.open((msw_img_pth))  # open the image with PIL
        msw_img = msw_img.convert("RGB")  # convert .bmp file to RGB image
        print("size of the converted RGB image is:", msw_img.size)

        # preprocess the converted RGB image
        box = (0, 0, 1024, 1024)
        msw_img = msw_img.crop(box)

        # convert RGB image to tensor
        msw_img = loader(msw_img).unsqueeze(0).to(device, torch.float)  

        # save the preprocessed msw images
        msw_img_list.append(msw_img)
    
    return msw_img_list


def visualize_recognition_results(prediction, counter, img, args):
    from omni.isaac.synthetic_utils import visualization
    from omni.isaac.shapenet import utils

    # setup visualization parameters
    img = torch.squeeze(img)
    counter = counter
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    score_thresh = 0.5
    mask_thresh = 0.5
    pred = prediction[0]
    # the shape of input image tensor is torch.Size([1, 3, 1024, 1024]), which represents the tensor includes 1 image, 3 channel(RGB), and the size of the image is 1024x1024 
    print("the shape of input image is:", img.shape)

    #the function permute(1,2,0) is used to change the size of the input image tensor to (1024, 1024, 3) for satisfying the requirments of imshow function in matplotlib
    np_image = img.permute(1, 2, 0).cpu().numpy()
    # np_image = img.cpu().numpy()


    # directory to save the visualized images to
    now = time.localtime()
    year = now.tm_year
    month = now.tm_mon
    day = now.tm_mday
    out_dir = os.path.join(os.getcwd(), "dps_ws/assests", "_out_recognized_imgs", f"{year}.{month}.{day}", "")
    os.makedirs(out_dir, exist_ok=True)

    for ax in axes:
        fig.suptitle(f"MSW Prediction Results ", fontsize=14)
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
    fig_name = os.path.join(out_dir, f"recognized_msw_image_{counter}.png")
    plt.savefig(fig_name)


def recognize(img, model, device):
    # assert img.shape == (1, 40, 40, 40)

    # move input to the GPU
    img = img.to(device)

    # forward pass
    with torch.no_grad():
        prediction = model(img)

    return prediction


if __name__ == "__main__":
    #parse training-related args
    parser = argparse.ArgumentParser("MSW Recognition")
    parser.add_argument(
        "--root",
        type=str,
        default="/media/walker2/Elements SE/DPS/dataset/can_bottle",
        help="Root directory containing real-world msw images.",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", required=True, help="List of WasteNet categories to use (space seperated)."
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize recognition results.")
    args, unknown_args = parser.parse_known_args()

    # start msw recognition
    main(args)
