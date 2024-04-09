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

        true_data = [input_data[i] for i in range(
            len(input_data))] + [true_data[i] for i in range(len(true_data))]
        pred_data = [input_data[i] for i in range(
            len(input_data))] + [pred_data[i] for i in range(len(pred_data))]

        ground_truth.append(true_data)
        prediction.append(pred_data)
    
    assert ground_truth[0][1] == ground_truth[1][0]
    return ground_truth, prediction

def generateGIF(symbol: str, data_path: str, cumprod: bool = False):
    # get npy file for pred and true
    ground_truth, prediction = recreateData(data_path, cumprod)

    image_path = data_path + "images/"

    # remove dir before creating
    if os.path.exists(image_path):
        os.system("rm -rf " + image_path)
    os.makedirs(image_path)

    interval = 10
    start = int(len(ground_truth) * 0.0) + interval
    end = int(len(ground_truth) * 0.1)

    # Create a list to store the images
    images = []

    # Loop through each index with tqdm progress bar
    for index in tqdm(range(start, end, interval)):

        plt.figure(figsize=(10, 3))

        # Plot the true and predicted values for the current index
        plt.plot(ground_truth[index], label="Ground Truth", linewidth=1, color="blue")

        for i in range(interval):
            if i == 1:
                plt.plot(prediction[index - i], label=f"Prediction", linewidth=1, color="orange")
            else:
                plt.plot(prediction[index - i], linewidth=1, label='_nolegend_', color="orange")

        if cumprod:
            plt.title(f"{symbol} Close Price(Data from Cumulative Percentage)")
        else:
            plt.title(f"{symbol} Close Price")
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

def getSymbol(model_id: str):
    return model_id.split("_")[1]

def getModelId(symbol: str, cumprod: bool = False):
    if cumprod:
        return f"PatchTST_{symbol}_pct_336_96"
    else:
        return f"PatchTST_{symbol}_336_96"

def isCumulative(model_id: str):
    return "pct" in model_id

symbols = [
    # "AAPL",
    # "MSFT",
    # "JPM",
    # "BAC",
    # "KO",
    "PG",
    # "JNJ",
    # "PFE",
    # "XOM",
    # "CVX",
]

# for symbol in symbols:
#     for cumprod in [ False]:
#         model_id = getModelId(symbol, cumprod)
#         data_path = getDataPath(model_id)
#         generateGIF(symbol, data_path, cumprod)
#         print(f"Generated GIF for {model_id}")

model_id = "PatchTST_AAPL_with_sentiment_336_96"
data_path = getDataPath(model_id)
generateGIF("AAPL", data_path, False)
print(f"Generated GIF for {model_id}")