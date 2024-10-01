import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
from gmodetector_py import Hypercube

# In your custom_unpickler function
def custom_unpickler(file_path):
    try:
        return pickle.load(file_path)
    except Exception as e:
        print(f"Standard loading failed due to {e}. Trying with map_location...")
        return torch.load(file_path, map_location=torch.device('cpu'))


def batch_inference(directory, school_pickle, method):
    """
    Run batch inference on a directory of hyperspectral images using a CubeSchool object and method.
    
    Parameters:
        directory (str): Path to the directory containing the hyperspectral images.
        school_pickle (str): Path to the pickled CubeSchool object.
        method (str): The inference method to use (e.g., "RF", "PCA").
    """
    
    # Load the CubeSchool object from the pickle file
    with open(school_pickle, 'rb') as f:
        school = custom_unpickler(f)



    # Check if the method exists in the school's learner_dict
    if method not in school.learner_dict:
        print(f"Method {method} not found in the provided CubeSchool.")
        return

    learner = school.learner_dict[method]

    # Loop through each file in the directory
    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files]

    for filename in files:
        # Check if the file is a hyperspectral image (assuming .raw extension)
        if filename.endswith("_Broadband.hdr"):
            # Load the hyperspectral image into a numpy array
            print("Loading img " + filename)
            # Load the hypercube
            hypercube_data = Hypercube(filename, min_desired_wavelength=400, max_desired_wavelength=1000)

            # Run the inference
            inference_map = learner.infer(hypercube_data.hypercube)

            # Generate the output filename based on the original file
            output_filename = filename.replace("_Broadband.hdr", "_segment_uncropped.png")

            # Save the inference_map
            plt.imsave(os.path.join(directory, output_filename), inference_map, cmap='jet')

            print(f"Inference completed for {filename}. Results saved as {output_filename}.")

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Run batch inference on a folder of hyperspectral images.")
    
    # Add command-line arguments
    parser.add_argument('--dir', type=str, help='Path to the directory containing the hyperspectral images.', required=True)
    parser.add_argument('--pickle', type=str, help='Path to the pickled CubeSchool object.', required=True)
    parser.add_argument('--method', type=str, help='The inference method to use (e.g., "RF", "PCA").', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the batch inference
    batch_inference(args.dir, args.pickle, args.method)

