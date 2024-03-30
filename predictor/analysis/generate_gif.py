# GIF Generation for NON Cumulative Percent changes

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def generateGIF(model_id: str, data_path: str):
    # get npy file for pred and true
    true = np.load(data_path + "true.npy")
    pred = np.load(data_path + "pred.npy")
    input = np.load(data_path + "x.npy")

    image_path = data_path + "images/"

    # remove dir before creating
    if os.path.exists(image_path):
        os.system("rm -rf " + image_path)
    os.makedirs(image_path)

    start = int(len(true) * 0.0)
    end = int(len(true) * 1.0)
    interval = 5

    # Loop through each index
    for index in range(start, end, interval):
        input_data = input[index, :, 4]
        true_data = true[index, :, 4]
        pred_data = pred[index, :, 4]

        true_data = [input_data[i] for i in range(
            len(input_data))] + [true_data[i] for i in range(len(true_data))]
        pred_data = [input_data[i] for i in range(
            len(input_data))] + [pred_data[i] for i in range(len(pred_data))]

        plt.figure()

        # Plot the true and predicted values for the current index
        plt.plot(true_data, label="True", linewidth=1)
        plt.plot(pred_data, label="Predicted", linewidth=1)

        plt.axvline(x=len(input_data), color='r', linestyle='--')

        plt.title(f"Index: {index}")

        plt.legend(loc="upper left")

        # Save the figure as an image
        plt.savefig(image_path + f'index_{index}.png')

        # Close the figure
        plt.close()

    # Create a list to store the images
    images = []

    # Loop through each index and open the corresponding image
    for index in range(start, end, interval):
        image = Image.open(image_path + f'index_{index}.png')
        images.append(image)

    # Save the images as a GIF
    images[0].save(data_path + 'output.gif', save_all=True,
                   append_images=images[1:], optimize=False, duration=200, loop=0)

    if os.path.exists(image_path):
        os.system("rm -rf " + image_path)


model_id = "AAPL_336_96_PatchTST_custom_ftM_sl336_ll48_pl96_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0"
data_path = "../results/" + model_id + "/data/"
generateGIF(model_id, data_path)
