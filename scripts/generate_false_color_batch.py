import os
import glob
from gmodetector_py import Hypercube, ImageChannel, FalseColor
import argparse

def generate_false_color_images(directory_path, green_cap, red_cap, blue_cap, min_wavelength, max_wavelength):
    """
    This function reads .hdr files from a directory, generates false color images 
    for each file and saves these images in the same directory with a '_falsecolor.png' suffix.

    Args:
    directory_path (str): Path to the directory containing the .hdr files.
    green_cap (float): Cap for green channel.
    red_cap (float): Cap for red channel.
    blue_cap (float): Cap for blue channel.
    min_wavelength (float): Minimum desired wavelength for the hypercube.
    max_wavelength (float): Maximum desired wavelength for the hypercube.

    Returns:
    None
    """

    # Assert directory exists
    assert os.path.exists(directory_path), f"Directory not found: {directory_path}"
    
    # Display the directory being processed
    print(f"Processing directory: {directory_path}")

    # Collect .hdr files
    hdr_files = glob.glob(os.path.join(directory_path, '*.hdr'))
    
    # Assert there are .hdr files in the directory
    assert hdr_files, f"No .hdr files found in {directory_path}"

    # Iterate over all .hdr files in the directory
    for file_path in hdr_files:
        # Display the current file being processed
        print(f"Processing file: {file_path}")
        
        # Load the hypercube
        test_cube = Hypercube(file_path, min_wavelength, max_wavelength)
        
        # Assert Hypercube object is created
        assert isinstance(test_cube, Hypercube), "Failed to create a Hypercube instance"

        # Create false color image
        stacked_component_image = FalseColor([
            ImageChannel(hypercube=test_cube, desired_component_or_wavelength="533.7419", color='green', cap=green_cap),
            ImageChannel(hypercube=test_cube, desired_component_or_wavelength="563.8288", color='red', cap=red_cap),
            ImageChannel(hypercube=test_cube, desired_component_or_wavelength="500.0404", color='blue', cap=blue_cap)
        ])
        
        # Assert FalseColor image is created
        assert isinstance(stacked_component_image, FalseColor), "Failed to create a FalseColor instance"

        # Save the image
        save_path = file_path.replace('.hdr', '_falsecolor.png')
        stacked_component_image.save(save_path)
        
        # Confirm image save
        print(f"Saved false color image at: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate false color images from .hdr files in a directory.")
    parser.add_argument('--directory_path', type = str, required=True,
                        help = "Path to the directory containing the .hdr files.")
    parser.add_argument('--green_cap', type = float, default=563,
                        help = "Cap for green channel")
    parser.add_argument('--red_cap', type = float, default=904,
                        help = "Cap for red channel")
    parser.add_argument('--blue_cap', type = float, default=406,
                        help = "Cap for blue channel")
    parser.add_argument('--min_wavelength', type = float, default=400,
                        help = "Minimum desired wavelength for the hypercube")
    parser.add_argument('--max_wavelength', type = float, default=1000,
                        help = "Maximum desired wavelength for the hypercube")
    args = parser.parse_args()

    # Display the parsed arguments
    print(f"Arguments: {vars(args)}")
    
    # Assert the directory path is provided
    assert args.directory_path, "No directory path provided"

    generate_false_color_images(args.directory_path, args.green_cap, args.red_cap, args.blue_cap, args.min_wavelength, args.max_wavelength)