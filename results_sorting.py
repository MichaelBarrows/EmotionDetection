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

# import_results()
# parameters:
#   folder : string - the folder for results to be imported from
# returns:
#   None
# description:
#   This function gets the output results files, loops over them, sorts them,
#       adds the first sorted row (highest f-score) to a new dataframe and
#       stores that dataframe.
def import_results (folder):
    new_results = []
    files = get_results_filenames(folder)
    for file in files:
        print("---" + file + "---")
        emotion = file.split("_")[0]
        if emotion == "best":
            continue
        results_df = helpers.load_dataset(folder + file)
        results_df = results_df.sort_values(['macro_avg_f1-score'],ascending=False)
        results_df = results_df.reset_index(drop=True)
        helpers.dataframe_to_csv(results_df, folder + file)
        for index, row in results_df.iterrows():
            new_results.append([emotion,
                                row.algorithm,
                                row.hyperparameter,
                                row.weighted_avg_precision,
                                row.weighted_avg_recall,
                                row["weighted_avg_f1-score"],
                                row.accuracy,
                                row.experiment_type,
                                row.metric_dump_id,
                                row.macro_avg_precision,
                                row.macro_avg_recall,
                                row["macro_avg_f1-score"],
                                row[emotion + "_precision"],
                                row[emotion + "_recall"],
                                row[emotion + "_f1-score"],
                                row["no_" + emotion + "_precision"],
                                row["no_" + emotion + "_recall"],
                                row["no_" + emotion + "_f1-score"]])
            break
    columns = ["emotion",
            "algorithm",
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
    new_results_df = pd.DataFrame(new_results, columns=columns)
    helpers.dataframe_to_csv(new_results_df, folder + "best_result_per_emotion.csv")

# import import_best_results_and_sort()
# parameters:
#   folder : string - the folder path that the files are stored in
# returns:
#   None
# description:
#   This function imports the best results file output in the previous function
#       as a dataframe, sorts it by highest f-score and stores the sorted
#       dataframe
def import_best_results_and_sort (folder):
    best_results_df = helpers.load_dataset(folder + "best_result_per_emotion.csv")
    best_results_df = best_results_df.sort_values(['macro_avg_f1-score'],ascending=False)
    best_results_df = best_results_df.reset_index(drop=True)
    helpers.dataframe_to_csv(best_results_df, folder + "best_result_per_emotion_sorted.csv")
