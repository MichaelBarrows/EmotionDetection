import helpers
import dataset as ds
import pandas as pd

# get_results_filenames()
# parameters:
#   version_path : string - the file path where the results files are stored
# returns
#   files : list - a list containing all of the results filenames
# description:
#   This function calls the path-fetcher function in the helpers file to
#       retrieve the filenames of the results files within that path. These
#       file names are then returned
def get_results_filenames (version_path):
    files = helpers.path_fetcher(version_path)
    return files

# algorithm_single_list()
# parameters:
#   algorithm_list : list - list containing algorithm names
# returns:
#   single_algorithm_list : list - list without repeating algorithm names
# description:
#   This functionn loops over a list of algorithm names, and adds them to a new
#       list - if they are not already in that list or are not "Decision Tree".
def algorithm_single_list (algorithm_list):
    single_algorithm_list = []
    for algorithm in algorithm_list:
        if algorithm not in single_algorithm_list and algorithm != "Decision Tree":
            single_algorithm_list.append(algorithm)
    return single_algorithm_list

# next_experiments()
# parameters:
#   mpt : int - the match percentage threshold
#   algorithm : string - the algorithm name
#   best_hyperparameter : string/int/float - The hyperparameter for the
#       algorithm that perfomed best.
#   experiments : list - a list for the experiments to be stored in
# returns:
#   experiments : list - a list of the experiments to be executed
# description:
#   This function takes the best performing hyperparameter for a given algorithm
#       and uses this to determine the experiments that should be run next.
#       This is only in relation to Linear SVM and Naive Bayes. The next
#       experiments are added to a global list.
def next_experiments (emotion, algorithm, best_hyperparameter, experiments):
    if algorithm == "KNN":
        experiments.append([emotion, algorithm, 1])
        experiments.append([emotion, algorithm, 9])
        experiments.append([emotion, algorithm, 11])
    if algorithm == "Linear SVM" or algorithm == "Naive Bayes":
        if best_hyperparameter == "default":
            experiments.append([emotion, algorithm, 0.3])
            experiments.append([emotion, algorithm, 0.5])
            experiments.append([emotion, algorithm, 1.5])
            experiments.append([emotion, algorithm, 1.8])
        else:
            if float(best_hyperparameter) > 1:
                experiments.append([emotion, algorithm, 1.5])
                experiments.append([emotion, algorithm, 1.8])
                experiments.append([emotion, algorithm, 0.5])
            elif float(best_hyperparameter) < 1:
                experiments.append([emotion, algorithm, 0.5])
                experiments.append([emotion, algorithm, 0.3])
                experiments.append([emotion, algorithm, 1.5])
    if algorithm == "Random Forest":
        experiments.append([emotion, algorithm, "100, 200"])
        experiments.append([emotion, algorithm, "150, 300"])
        experiments.append([emotion, algorithm, "200, 400"])
        experiments.append([emotion, algorithm, "300, 600"])
    return experiments

# get_existing_results()
# parameters:
#   folder : str - the path for which the relevant results are stored in
#   dataset_type : str - used for storing CSV file
#   n_grams : string - a string detailing the n-grams for storage location
# returns:
#   None
# description:
#   This function gets the results files, loops over the list containing them,
#       imports them as a dataframe, sorts and groups them. The list of all
#       algorithms is then iterated over, with the best performing
#       hyperparameter sent to next_experiments() to decide which experiments
#       should be performed next. Finally, the next experiments are converted
#       to a DataFrame and stored in a CSV file.
def get_existing_results (folder, n_grams):
    experiments = []
    for file in get_results_filenames(folder):
        emotion = file.split("_")[0]
        if emotion == "best":
            continue
        results_df = helpers.load_dataset(folder + file)
        results_df = results_df.sort_values(['macro_avg_f1-score'],ascending=False).groupby('algorithm').head(3)
        results_df = results_df.reset_index(drop=True)
        algorithms = algorithm_single_list(results_df.algorithm.tolist())
        for algorithm in algorithms:
            relevant_rows = results_df[results_df.algorithm == algorithm]
            for index, row in relevant_rows.iterrows():
                experiments = next_experiments(emotion, algorithm, row.hyperparameter, experiments)
                break
    new_experiments_df = pd.DataFrame(experiments, columns=["emotion", "algorithm", "hyperparameter"])
    helpers.dataframe_to_csv(new_experiments_df, "/home/michael/MRes/actual_project/emotion_detection/" + n_grams + "/next_experiments.csv")

def run (folder, n_grams):
    get_existing_results (folder, n_grams)
