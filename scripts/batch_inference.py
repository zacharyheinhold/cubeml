import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
from gmodetector_py import Hypercube, ImageChannel, FalseColor
from cubeml.model_evaluation import false_color_image
from cubeml import CubeLearner
from cubeml import CubeSchool

def new_infer_placeholder(self, *args, **kwargs):
    # Placeholder method, the actual implementation is not needed here
    pass

CubeLearner.new_infer = new_infer_placeholder

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'new_infer':
            # Return a dummy function for 'new_infer'
            return lambda self, *args, **kwargs: None
        return super().find_class(module, name)

def load_cubelearner_state(file_prefix, save_dir="./"):
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    state_path = os.path.join(save_dir, f"{file_prefix}_state.pkl")
    
    # Load the saved model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the rest of the learner's state from pickle
    with open(state_path, 'rb') as f:
        learner = pickle.load(f)

    # Attach the model to the learner
    learner.model = model

    return learner

def load_data(file_path):
    # Extract the directory and file prefix
    file_dir = os.path.dirname(file_path)
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == '.pt':
        # Load CubeLearner state for .pt files
        return load_cubelearner_state(file_prefix, file_dir)
    elif file_extension == '.pkl':
        # Loading pickle file with custom unpickler
        with open(file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()

        if isinstance(data, CubeSchool):
            # Handle CubeSchool object
            return data
        elif isinstance(data, CubeLearner):
            # Handle CubeLearner object
            return data
        else:
            raise ValueError("Unsupported pickle object type.")
    else:
        raise ValueError("Unsupported file type.")

    return data


        
def infer(learner, hypercube_data, batch_size=256):
    # Flatten the 3D hypercube into 2D so we can run our classifier on it
    print("Current version sanity test1")
    num_rows, num_cols, num_bands = hypercube_data.shape
    flattened_data = hypercube_data.reshape(num_rows * num_cols, num_bands)

    # Initialize the inference map
    inference_map = np.zeros((num_rows * num_cols,))

    # Check if the model is a TransformerNN
    if hasattr(learner, 'model_type') and learner.model_type == "TNN":
        print("Current version sanity test2")
        #if learner.pos_enc is None:
        #	learner.pos_enc = get_positional_encoding(seq_len=learner.n_input_features, d_model=learner.d_model)

        # Loop over the flattened data in chunks
        for start_idx in range(0, len(flattened_data), batch_size):
            end_idx = min(start_idx + batch_size, len(flattened_data))
            chunk_data = flattened_data[start_idx:end_idx]
            chunk_tensor = torch.tensor(chunk_data).to(learner.device)

            # Move the model to the same device if it's not already
            learner.model.to(learner.device)

            # Make predictions for the current chunk
            with torch.no_grad():
                chunk_pred = learner.model(chunk_tensor, learner.model.pos_enc.to(learner.device))
                chunk_pred = torch.argmax(chunk_pred, dim=1).cpu().numpy()

            # Fill the corresponding section of the inference map with the chunk predictions
            inference_map[start_idx:end_idx] = chunk_pred

    else:
        # For non-TransformerNN models, use the model's predict method directly on the flattened data
        inference_map = learner.model.predict(flattened_data)

    # Reshape the predictions back into the 2D spatial configuration
    inference_map = inference_map.reshape(num_rows, num_cols)

    # Return the predictions as a 2D array
    return inference_map

def batch_inference(directory, data_file, method, string_to_exclude, false_color, green_cap, red_cap, blue_cap, min_wavelength, max_wavelength, green_wavelength, red_wavelength, blue_wavelength, use_quantization=False):
    
    data = load_data(data_file)
    if isinstance(data, CubeSchool):
        # Extract CubeLearner from CubeSchool
        learner = data.learner_dict.get(method)
        if learner is None:
            print(f"Method {method} not found in the provided CubeSchool.")
            return
    elif isinstance(data, CubeLearner):
        learner = data
    else:
        learner = data
        
#     # Check CubeLearner attributes
#     print("Model Type:", learner.model_type)
#     print("Device:", learner.device)
#     print("Number of Classes:", learner.num_classes)

#     # Check if the model is correctly loaded
#     if learner.model is not None:
#         # Check TransformerNN model attributes
#         print("Model d_model:", getattr(learner.model, 'd_model', None))
#         print("Model use_embedding:", getattr(learner.model, 'use_embedding', None))
#         print("Model num_classes:", getattr(learner, 'num_classes', None))
#         print("Positional Encoding (pos_enc):", getattr(learner.model, 'pos_enc', None))
#     else:
#         print("Model not loaded correctly.")

#     # Additional check to ensure the model is of the expected type
#     if isinstance(learner.model, CubeLearner.TransformerNN):
#         print("Model is a TransformerNN.")
#     else:
#         print("Model is not a TransformerNN.")
        
    learner.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files if file.endswith("_Broadband.hdr") and (string_to_exclude is None or string_to_exclude not in file)]
    
    if use_quantization:
        learner.model = torch.quantization.quantize_dynamic(
        learner.model, {nn.Linear, nn.MultiheadAttention}, dtype=torch.qint8)

    for filename in files:
        
        #if filename == files[0]:
        #    prof = torch.profiler.profile(with_stack=True, profile_memory=True, record_shapes=True)
        #    prof.start()
        #if filename.endswith("_Broadband.hdr"):
        print("Loading img " + filename)
        hypercube_data = Hypercube(filename, min_desired_wavelength=min_wavelength, max_desired_wavelength=max_wavelength)

        #inference_map = learner.infer(hypercube_data.hypercube)
        inference_map = infer(learner, hypercube_data.hypercube)


        output_filename = filename.replace("_Broadband.hdr", "_segment_uncropped_processed.png")
        output_filename = os.path.join(directory, output_filename)

        # Generate and save the false color image from the segmentation map
        seg_false_color = false_color_image(predictions=inference_map, colors=learner.colors)
        seg_false_color.save(output_filename)

        if false_color:
            # Generate and save the false color image from the original hypercube
            print("Saving out false color RGB in addition to segmentation mask")
            rgb_false_color = FalseColor([
                ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=green_wavelength, color='green', cap=green_cap),
                ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=red_wavelength, color='red', cap=red_cap),
                ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=blue_wavelength, color='blue', cap=blue_cap)
            ])

            rgb_false_color.image = rgb_false_color.image.rotate(-90, expand=True)

            rgb_false_color.save(os.path.basename(output_filename.replace("_segment_uncropped_processed.png",
                                                                          "_rgb_processed.png")),
                                 output_dir = directory)

        print(f"Inference completed for {filename}. Results saved as {output_filename}.")
        #if filename == files[0]:
        #    prof.stop()
        #    with open("profiler_output.txt", "w") as f:
        #        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on a folder of hyperspectral images.")
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing hyperspectral images.')
    parser.add_argument('--pickle', type=str, required=True,
                        help='Path to the pickled CubeSchool object.')
    parser.add_argument('--method', type=str, required=True,
                        help='Inference method (e.g., "RF", "PCA").')
    parser.add_argument('--string_to_exclude', type=str, default=None,
                        help='String to identify files to be excluded from processing.')
    parser.add_argument('--false_color', action='store_true',
                        help='Generate false color images.')
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
    parser.add_argument('--green_wavelength', type=str, default="533.7419",
                        help='Wavelength for green channel.')
    parser.add_argument('--red_wavelength', type=str, default="563.8288",
                        help='Wavelength for red channel.')
    parser.add_argument('--blue_wavelength', type=str, default="500.0404",
                        help='Wavelength for blue channel.')
    parser.add_argument('--use_quantization', action='store_true',
                        help='Enable quantization for faster inference on CPU.')

    args = parser.parse_args()

    batch_inference(directory=args.dir,
                    data_file=args.pickle,
                    method=args.method,
                    string_to_exclude=args.string_to_exclude,
                    false_color=args.false_color,
                    green_cap=args.green_cap,
                    red_cap=args.red_cap,
                    blue_cap=args.blue_cap,
                    green_wavelength=args.green_wavelength,
                    red_wavelength=args.red_wavelength,
                    blue_wavelength=args.blue_wavelength,
                    min_wavelength=args.min_wavelength,
                    max_wavelength=args.max_wavelength,
                    use_quantization=args.use_quantization
                    )

