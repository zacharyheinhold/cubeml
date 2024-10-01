from cubeml import CubeLearner
import os
import matplotlib.pyplot as plt

class CubeSchool:
    def __init__(self, training_data, model_types, colors, data_types=["Train", "Test"]):
        self.training_data = training_data
        self.model_types = model_types
        self.colors = colors
        self.data_types = data_types

        self.learner_dict = {}
        self.cfx_dict = {data_type: {} for data_type in data_types}
        self.iou_dict = {data_type: {model_type: [] for model_type in model_types} for data_type in data_types}

    def fit_models(self):
        for model_type in self.model_types:
            # Instantiate CubeLearner and store in dictionary
            self.learner_dict[model_type] = CubeLearner(self.training_data, model_type=model_type, colors=self.colors, train_test = True if model_type != "PCA" else False)
            
            # Fit the model
            if model_type in ["RF"]:
                self.learner_dict[model_type].fit(max_depth = 23,
                                                  criterion = 'entropy',
                                                  max_features = 'sqrt',
                                                  min_samples_leaf = 8,
                                                  min_samples_split = 41,
                                                  n_estimators = 163)
            elif model_type in ["DTC"]:
                self.learner_dict[model_type].fit(max_depth = 23,
                                                  criterion = 'entropy',
                                                  max_features = 'sqrt',
                                                  min_samples_leaf = 8,
                                                  min_samples_split = 41)
            else: 
                self.learner_dict[model_type].fit()

    def analyze_models(self):
        for model_type in self.model_types:
            for data_type in self.data_types:
                # Plot the results
                self.learner_dict[model_type].visualize(data_type)
                
                if model_type != "PCA":  # PCA does not have a confusion matrix or IoU
                    # Calculate and store cfx and iou
                    self.learner_dict[model_type].plot_cfx(data_type)
                    self.cfx_dict[data_type][model_type] = self.learner_dict[model_type].cfx_train if data_type == "Train" else self.learner_dict[model_type].cfx_test  # Update cfx for the corresponding data_type
                    
                    # Assigning the iou_dict_list to the appropriate key in the iou_dict dictionary
                    self.iou_dict[data_type][model_type] = self.learner_dict[model_type].iou_dict_train if data_type == "Train" else self.learner_dict[model_type].iou_dict_test
                    
    def multi_plot(self, 
                   data_types: list = ["Train", "Test"], 
                   output_dir: str = "output/learner_plots/",
                   verbose: bool = False):
        """
        Run plot_cfx and visualize, display, and save the outputs for all learners.

        Parameters
        ----------
        data_types : list, optional
            The types of data to process. Defaults to ["Train", "Test"].
        output_dir : str, optional
            The directory to save the plots to. Defaults to "learner_outputs/".
        verbose : bool, optional
            Whether to print processing information. Defaults to False.
        """

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through the learners
        for model_type, learner in self.learner_dict.items():
            for data_type in data_types:
                if verbose:
                    print(f"Processing model type: {model_type}, data_type: {data_type}")

                if model_type != "PCA":
                    # Plot cfx
                    if verbose:
                        print(f"Plotting cfx for {model_type} - {data_type}")
                    learner.plot_cfx(data_type)

                    # Save the plot
                    plt.savefig(f"{output_dir}{model_type}_{data_type}_cfx_heatmap.png")
                    plt.close()  # Close the plot

                # Visualize
                if verbose:
                    print(f"Visualizing for {model_type} - {data_type}")
                learner.visualize(data_type)

                # Save the visualization
                plt.savefig(f"{output_dir}{model_type}_{data_type}_plot.png")
                plt.close()  # Close the plot

    def save_cubelearners_from_cubeschool(self, file_prefix, save_dir="./"):
        for key, learner in self.learner_dict.items():
            if learner.model_type == 'TNN':
                continue  # Skip TNN models
    
            # Define the filename for the pkl file
            model_filename = os.path.join(save_dir, f"{file_prefix}_{key}.pkl")
    
            # Save the entire CubeLearner object to a pkl file
            with open(model_filename, 'wb') as f:
                pickle.dump(learner, f)

    def run(self):
        self.fit_models()
        self.analyze_models()

