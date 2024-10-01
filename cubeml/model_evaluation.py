import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from gmodetector_py import Hypercube
from cubeml import CubeLearner
from typing import Union, List



def presentable_table(labels_key_df, iou_dict):
    iou_list = []
    for data_type, methods in iou_dict.items():
        for method, label_iou in methods.items():
            if isinstance(label_iou, dict):  
                for label, iou in label_iou.items():
                    iou_list.append((data_type, method, str(label), iou))

    iou_df = pd.DataFrame(iou_list, columns=['DataType', 'Method', 'Label', 'IOU'])
    
    # Convert 'Integer' in labels_key_df to str for the mapping
    labels_key_df['Integer'] = labels_key_df['Integer'].astype(str)
    
    # Map 'Label' in iou_df using 'Integer' in labels_key_df
    label_map = dict(zip(labels_key_df['Integer'], labels_key_df['Label']))
    iou_df['Label'] = iou_df['Label'].map(label_map)
    
    # Merge with labels_key_df
    iou_df = pd.merge(iou_df, labels_key_df, left_on='Label', right_on='Label')
    
    # Pivot the DataFrame to have the desired format
    table = iou_df.pivot_table(index=['Method', 'DataType'], columns='Label', values='IOU')
    
    # Calculate the mean IoU for each method and data type and add as a new column
    table['mean IoU'] = table.apply(np.mean, axis=1)
    
    # Sort the DataFrame so the 'train' rows come before the 'test' rows for each method
    table.sort_values(['Method', 'DataType'], ascending=[True, False], inplace=True)
    
    return table

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def false_color_image(predictions, colors):
    # Make sure predictions are int
    predictions = predictions.astype(int)

    # Convert colors to RGB tuples
    color_tuples = [hex_to_rgb(color) for color in colors]

    # Create an empty 3D array with the shape of the prediction map in the first two dimensions and a depth of 3 for RGB
    color_map = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

    # Assign colors to the classes
    for i, color in enumerate(color_tuples):
        color_map[predictions == i] = color

    # Convert to PIL Image
    img = Image.fromarray(color_map)
    return img

def match_files(directory):
    # Initialize a dictionary to store data
    data_dict = {}

    # Iterate over files in directory
    for filename in os.listdir(directory):
        # Remove extension to get ID
        id_, ext = os.path.splitext(filename)
        ext = ext.lstrip('.')  # Remove leading dot from extension

        # Remove the suffixes like "_Broadband" or "_rgb" if present
        id_ = id_.split('_Broadband')[0].split('_rgb')[0]

        # Ensure ext is one of the expected file types
        if ext not in ["hdr", "raw", "jpg", "png", "csv"]:
            continue

        # If ID is new, add to data_dict with empty file type placeholders
        if id_ not in data_dict:
            data_dict[id_] = {"id": id_, "hdr": 'NA', "raw": 'NA', "jpg": 'NA', "png": 'NA', "csv": 'NA'}

        # Add full file path to appropriate file type under its ID
        data_dict[id_][ext] = os.path.join(directory, filename)

    # Convert to pandas DataFrame and return
    df = pd.DataFrame(list(data_dict.values()))
    return df

def multi_panel_figure(png_file, false_color_file, num_panels=3):
    # Open and rotate PNG file
    png = Image.open(png_file).rotate(270, expand=True)

    # Load false color image
    false_color = Image.open(false_color_file)

    # Ensure both images have same size
    if png.size != false_color.size:
        false_color = false_color.resize(png.size)

    # Create overlay of PNG and false color image
    overlay = Image.blend(png.convert('RGBA'), false_color.convert('RGBA'), alpha=0.5)

    # Merge all images together
    if num_panels == 3:
        images = [png, false_color, overlay]
    elif num_panels == 2:
        images = [false_color, overlay]
    else:
        raise ValueError("num_panels must be either 2 or 3")

    # Ensure all images are the same size
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create a blank canvas for the final image
    new_img = Image.new('RGBA', (total_width, max_height))

    # Paste each image into the final image
    for idx, img in enumerate(images):
        new_img.paste(img, (widths[idx] * idx, 0))

    # Return the final image
    return new_img

def generate_falsecolor_images(df: pd.DataFrame, 
                               learner_dict: dict, 
                               model_types: Union[str, List[str]], 
                               colors: dict, 
                               output_dir: str = "falsecolor/", 
                               min_wave: float = 400, 
                               max_wave: float = 1000):
    """
    Generate and save falsecolor images for given model types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the hdr file paths.
    learner_dict : dict
        Dictionary containing the models.
    model_types : str or List[str]
        The model types to use for inference.
    colors : dict
        Dictionary containing the colors to use for the false color image.
    output_dir : str, optional
        The directory to save the images to. Defaults to "falsecolor/".
    min_wave : float, optional
        The minimum desired wavelength. Defaults to 400.
    max_wave : float, optional
        The maximum desired wavelength. Defaults to 1000.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure model_types is a list even if it's a single string
    if isinstance(model_types, str):
        model_types = [model_types]
    
    # Loop over all hdr files in the DataFrame
    for _, row in df.iterrows():
        # Load the hyperspectral image cube
        hypercube = Hypercube(row['hdr'], min_desired_wavelength=min_wave, max_desired_wavelength=max_wave)
        
        for model_type in model_types:
            # Get the learner for the model type
            learner = learner_dict[model_type]
            
            # Make predictions
            predictions = learner.infer(hypercube.hypercube)
            
            # Generate the false color image
            img = false_color_image(predictions=predictions, colors=colors)

            # Generate filename from the ID and model_type
            filename = f"{output_dir}{row['id']}_{model_type}.png"

            # Save the image
            img.save(filename)


def compare_inferences(df: pd.DataFrame, 
                       model_types: list = ["LDA", "RF", "GBC", "ABC", "LR", "GNB", "DTC"],
                       falsecolor_dir: str = "output/falsecolor/",
                       output_dir: str = "output/panel_images/"):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        img_id = row['id']
        png = row['png']

        fig, axs = plt.subplots(len(model_types), 1, figsize=(15, len(model_types)*5))
        if len(model_types) == 1:  # Ensure there's only one axis if there's one model_type
            axs = [axs]

        for ax, model_type in zip(axs, model_types):
            print(f"Processing with {model_type} model")
            false_color_file = os.path.join(falsecolor_dir, f"{img_id}_{model_type}.png")
            img = multi_panel_figure(png, false_color_file, num_panels=3)

            ax.imshow(img)
            ax.axis('off')
            ax.text(-0.02, 0.5, model_type, size=36, ha="right", va="center", rotation='vertical', transform=ax.transAxes)

        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.savefig(os.path.join(output_dir, f"{img_id}_panel.png"), dpi=300)
        plt.close(fig)