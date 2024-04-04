# GIF Generation for NON Cumulative Percent changes

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

def recreateData(data_path: str, cumprod: bool = False):
    # get npy file for pred and true
    true = np.load(data_path + "true.npy")
    pred = np.load(data_path + "pred.npy")
    input = np.load(data_path + "x.npy")

    # we only need last column which is Close
    input = input[:, :, -1]
    true = true[:, :, -1]
    pred = pred[:, :, -1]

    print(input.shape)

    ground_truth = []
    prediction = []

    if cumprod:
        starting_inputs = np.cumprod(input[:, 0] + 1, axis=0)

    plt.plot(starting_inputs[:336])

    plt.savefig(data_path + "starting_inputs.png")
    
    for index in range(len(true)):
        if cumprod:
            input[index][0] = 0
            input_data = np.cumprod(input[index] + 1) * starting_inputs[index]
            true_data = np.cumprod(true[index] + 1) * input_data[-1]
            pred_data = np.cumprod(pred[index] + 1) * input_data[-1]
        else:
            input_data = input[index]
            true_data = true[index]
            pred_data = pred[index]

        print(input_data[0])

        true_data = [input_data[i] for i in range(
            len(input_data))] + [true_data[i] for i in range(len(true_data))]
        pred_data = [input_data[i] for i in range(
            len(input_data))] + [pred_data[i] for i in range(len(pred_data))]

        ground_truth.append(true_data)
        prediction.append(pred_data)
    
    assert ground_truth[0][1] == ground_truth[1][0]
    return ground_truth, prediction

def generateGIF(data_path: str, cumprod: bool = False):
    # get npy file for pred and true
    ground_truth, prediction = recreateData(data_path, cumprod)

    image_path = data_path + "images/"

    # remove dir before creating
    if os.path.exists(image_path):
        os.system("rm -rf " + image_path)
    os.makedirs(image_path)

    interval = 5
    start = int(len(ground_truth) * 0.0) + interval
    end = int(len(ground_truth) * 0.01)

    # Create a list to store the images
    images = []

    # Loop through each index with tqdm progress bar
    for index in tqdm(range(start, end, interval)):

        plt.figure()

        # Plot the true and predicted values for the current index
        plt.plot(ground_truth[index], label="True", linewidth=1)

        for i in range(interval):
            plt.plot(prediction[index - i], label=f"Pred {interval - i}", linewidth=1)

        plt.title(f"Index: {index}")
        plt.legend(loc="upper left")
        plt.axvline(x=336, color='r', linestyle='--')

        # Save the figure as an image
        plt.savefig(image_path + f'index_{index}.png', bbox_inches='tight')

        # Close the figure
        plt.close()

        # Open the corresponding image and append it to the list
        image = Image.open(image_path + f'index_{index}.png')
        images.append(image)

    # Save the images as a GIF
    images[0].save(data_path + 'output.gif', save_all=True,
                   append_images=images[1:], optimize=False, duration=200, loop=0)

    if os.path.exists(image_path):
        os.system("rm -rf " + image_path)


def getDataPath(model_id: str):
    return "./predictor/results/" + model_id + "/data/"


# model_id = "PatchTST_AAPL_336_96"
model_id = "PatchTST_AAPL_pct_336_96"
data_path = getDataPath(model_id)

# generateGIF(data_path, False)
generateGIF(data_path, True)
