def get_param_ranges(learner, model_type, lower_bound=0.8, upper_bound=1.2):
    """
    Derive parameter ranges based on pseudo-optimized parameters from an initial grid search.

    This function computes parameter ranges around pseudo-optimal values obtained 
    from an initial grid search using CubeLearner.fit(automl="grid"). The produced 
    ranges can then be used for more detailed optimization, such as genetic algorithms, 
    using CubeLearner.fit(automl="genetic").

    Parameters:
    - learner (CubeLearner): Instance of the CubeLearner class containing 
                             pseudo-optimal parameters and a model.
    - model_type (str): The model type for which to obtain parameter ranges.
    - lower_bound (float, optional): Lower percentage to compute the parameter range. Default is 0.8.
    - upper_bound (float, optional): Upper percentage to compute the parameter range. Default is 1.2.

    Returns:
    - dict: Parameter ranges constructed around the pseudo-optimal values.
    
    Example:
    ```python
    # Assuming 'my_learner' is an instance of CubeLearner for "RF" model type with optimal parameters set.
    my_learner.fit(automl="grid")
    param_ranges = get_param_ranges(my_learner, "RF")
    print(param_ranges)
    # Output might look like:
    # {'n_estimators': (80, 120), 'max_depth': [None], 'min_samples_split': (32, 48), ...}
    ```

    Note:
    This function is specifically tailored for CubeLearner's use and may not be generalizable
    for other frameworks or situations.
    """
    param_ranges = {}
    for k, v in learner.optimal_params.items():
        param_type = type(learner.model.get_params()[k])  # Getting the type of parameter from the CubeLearner's model

        if isinstance(v, int) and param_type in [int, np.int32, np.int64]:  # If parameter is integer
            param_ranges[k] = (int(lower_bound * v), int(upper_bound * v))  # adjust range as required
        elif isinstance(v, float) and param_type in [float, np.float32, np.float64]:  # If parameter is float
            param_ranges[k] = (lower_bound * v, upper_bound * v)  # adjust range as required
        elif isinstance(v, str) or v is None:  # If parameter is a string or None
            param_ranges[k] = [v]  # use list with single value

    return param_ranges

