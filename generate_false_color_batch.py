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

    # Iterate over all .hdr files in the directory
    for file_path in glob.glob(os.path.join(directory_path, '*.hdr')):
        # Load the hypercube
        test_cube = Hypercube(file_path, min_wavelength, max_wavelength)

        # Create false color image
        stacked_component_image = FalseColor([ImageChannel(hypercube = test_cube,
                                                           desired_component_or_wavelength = "533.7419",
                                                           color = 'green',
                                                           cap = green_cap),
                                              ImageChannel(hypercube = test_cube,
                                                           desired_component_or_wavelength = "563.8288",
                                                           color = 'red',
                                                           cap = red_cap),
                                              ImageChannel(hypercube = test_cube, 
                                                           desired_component_or_wavelength = "500.0404",
                                                           color = 'blue',
                                                           cap = blue_cap)])
        # Save the image
        print("File in path: " + file_path)
        file_out_path = file_path.replace('.hdr', '_falsecolor.png')
        file_out_path = file_out_path.replace('./', '/')
        print("Saving file to : " + file_out_path)
        stacked_component_image.save(file_out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate false color images from .hdr files in a directory.")
    parser.add_argument('--directory_path', type = str, required=True,
                        help = "Path to the directory containing the .hdr files.")
    parser.add_argument('--green_cap', type = float, default=563, #9_30_24 (2/3)*563
                        help = "Cap for green channel")
    parser.add_argument('--red_cap', type = float, default=904, #9_30_24 (2/3)*904
                        help = "Cap for red channel")
    parser.add_argument('--blue_cap', type = float, default=406, #9_30_24 (2/3)*406
                        help = "Cap for blue channel")
    parser.add_argument('--min_wavelength', type = float, default=400,
                        help = "Minimum desired wavelength for the hypercube")
    parser.add_argument('--max_wavelength', type = float, default=1000,
                        help = "Maximum desired wavelength for the hypercube")
    args = parser.parse_args()

    generate_false_color_images(args.directory_path, args.green_cap, args.red_cap, args.blue_cap, args.min_wavelength, args.max_wavelength)
