import helpers
import dataset as ds
import pandas as pd
import run

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

# get_first_experimental_results()
# parameters:
#   data : DataFrame - the data containing the data for processing
#   mpt : int - the match percentage threshold to be retained of the dataset
#   results_df : DataFrame - The dataframe holding the existing results
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function creates the first set of experimental results for each of the
#       four machine learning algorithms with hyperparameters by removing part
#       of the dataset with a words_matched_percentage value lower than the mpt
#       variable. The results are then stored in a list, converted (and
#       appended) to a dataframe and stored in a CSV file.
def get_first_experimental_results (data, emotion, results_df, n_grams):
    print("--- " + str(emotion) + " ---")
    results = []

    results.append(run.run_knn_classification(data, emotion,  3, "initial experiment", n_grams))
    results.append(run.run_knn_classification(data, emotion, 7, "initial experiment", n_grams))

    results.append(run.run_linear_svm_classification(data, emotion, 0.8, "initial experiment", n_grams))
    results.append(run.run_linear_svm_classification(data, emotion, 1.2, "initial experiment", n_grams))

    results.append(run.run_naive_bayes_classification(data, emotion, 0.8, "initial experiment", n_grams))
    results.append(run.run_naive_bayes_classification(data, emotion, 1.2, "initial experiment", n_grams))

    results.append(run.run_random_forest_classification(data, emotion, [50, 100], "initial experiment", n_grams))
    results.append(run.run_random_forest_classification(data, emotion, [75, 150], "initial experiment", n_grams))
    columns = ["algorithm",
            "hyperparameter",
            "weighted_avg_precision",
            "weighted_avg_recall",
            "weighted_avg_f1-score",
            "accuracy",
            "experiment_type",
            "metric_dump_id",
            "macro_avg_precision",
            "macro_avg_recall",
            "macro_avg_f1-score",
            emotion + "_precision",
            emotion + "_recall",
            emotion + "_f1-score",
            "no_" + emotion + "_precision",
            "no_" + emotion + "_recall",
            "no_" + emotion + "_f1-score"]
    results = pd.DataFrame(results, columns=columns)
    results_df = results_df.append(results)
    print(results_df[["algorithm", "hyperparameter", "macro_avg_f1-score", "accuracy", emotion + "_f1-score", "no_" + emotion + "_f1-score"]])
    return results_df

# negation_not_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function get the filenames of existing results files and iterates over
#       the list. The Twitter data is loaded and for each results file
#       (different MPT value), the results file is loaded to a dataframe and
#       get_first_experimental_results() is called to generate the results.
#       the new results are stored.
def execute (folder, n_grams):
    results_files = get_results_filenames(folder)
    data = helpers.load_dataset(ds.dataset + ds.file)
    for results_file in results_files:
        emotion = results_file.split("_")[0]
        if emotion == "best":
            continue
        results_df = helpers.load_dataset(folder + results_file)
        results_df = get_first_experimental_results(data, emotion, results_df, n_grams)
        helpers.dataframe_to_csv(results_df, folder + results_file)
