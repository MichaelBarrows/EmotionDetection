import helpers
import dataset as ds
import run
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

# process_experiments()
# parameters:
#   data : DataFrame - dataframe containing the twitter data
#   mpt : int - the match percentage threshold for the data`
#   experiments : DataFrame - dataframe containing details of experiments to be
#       conducted
#   results_df : DataFrame - dataframe containing existing results
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   results_df : DataFrame - dataframe containing existing and new results
# description:
#   This function retains the data above the mpt threshold and iterates over the
#       experiments dataframe. For each row in the experiments df, it executes
#       the given experiment (and modifies the hyperparameter for random forest)
#       and stores the results. The results are then added to the results df,
#       which is then returned.
def process_experiments (data, emotion, experiments, results_df, n_grams):
    results = []
    counter = 1
    for index, row in experiments.iterrows():
        print(str(emotion) + "- " + str(counter) + " / " + str(len(experiments)) + " " + row.algorithm + " - " + str(row.hyperparameter))
        counter += 1
        if row.algorithm == "KNN":
            results.append(run.run_knn_classification(data, emotion, row.hyperparameter, "experiment", n_grams))
        elif row.algorithm == "Linear SVM":
            results.append(run.run_linear_svm_classification(data, emotion, row.hyperparameter, "experiment", n_grams))
        elif row.algorithm == "Naive Bayes":
            results.append(run.run_naive_bayes_classification(data, emotion, row.hyperparameter, "experiment", n_grams))
        elif row.algorithm == "Random Forest":
            hyperparameter = row.hyperparameter.split(', ')
            results.append(run.run_random_forest_classification(data, emotion, [int(hyperparameter[0]), int(hyperparameter[1])], "experiment", n_grams))
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
    results_df = results_df.reset_index(drop=True)
    print(results_df[["algorithm", "hyperparameter", "macro_avg_f1-score", "accuracy", emotion + "_f1-score", "no_" + emotion + "_f1-score"]])
    return results_df

# process_negation_not_handled_experiments()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function gets the filenames for the results files, imports the twitter
#       data and the experiments data. The results files are looped over, with
#       the results data for the given MPT imported and experiments within that
#       MPT are retained. The process_experiments() function is then called to
#       execute the experiments, and the results data is stored.
def execute (folder, n_grams):
    results_files = get_results_filenames(folder)
    data = helpers.load_dataset(ds.dataset + ds.file)
    experiments_df = helpers.load_dataset("/home/michael/MRes/actual_project/emotion_detection/" + n_grams + "/next_experiments.csv")
    for results_filename in results_files:
        emotion = results_filename.split("_")[0]
        if emotion == "best":
            continue
        experiments = experiments_df[experiments_df.emotion == emotion]
        results_df = helpers.load_dataset(folder + results_filename)
        results_df = process_experiments(data, emotion, experiments, results_df, n_grams)
        helpers.dataframe_to_csv(results_df, folder + results_filename)
    return
