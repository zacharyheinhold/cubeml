def count_duplicates(features_list):
    """
    Count duplicate spectra in the features list.

    Parameters:
    features_list (list of ndarray): list of 1D arrays representing spectra

    Returns:
    count_duplicates (int): count of duplicate spectra
    """
    features_dict = {}
    duplicates_count = 0

    for spectrum in features_list:
        spectrum_hash = hash(spectrum.tobytes())
        if spectrum_hash in features_dict:
            duplicates_count += 1
        else:
            features_dict[spectrum_hash] = spectrum

    return duplicates_count


def remove_duplicates(features_list):
    """
    Remove duplicate spectra from the features list.

    Parameters:
    features_list (list of ndarray): list of 1D arrays representing spectra

    Returns:
    unique_features_list (list of ndarray): list of 1D arrays representing unique spectra
    """
    features_dict = {}

    for spectrum in features_list:
        spectrum_hash = hash(spectrum.tobytes())
        features_dict[spectrum_hash] = spectrum

    unique_features_list = list(features_dict.values())
    return unique_features_list


def process_duplicates(features_list):
    """
    Count and remove duplicate spectra from the features list.

    Parameters:
    features_list (list of ndarray): list of 1D arrays representing spectra

    Returns:
    unique_features_list (list of ndarray): list of 1D arrays representing unique spectra
    count_duplicates (int): count of duplicate spectra
    """
    count = count_duplicates(features_list)
    unique_features = remove_duplicates(features_list)
    
    return unique_features, count
