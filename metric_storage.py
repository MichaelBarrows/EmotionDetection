import helpers
import dataset as ds
import pandas as pd

# get_metric_storage_filenames()
# parameters:
#   None
# returns
#   files : list - a list containing all of the metric files
# description:
#   This function calls the path-fetcher function in the helpers file to
#       retrieve the filenames of the results files within that path. These
#       file names are then returned
def get_metric_storage_filenames ():
    files = helpers.path_fetcher(ds.metric_storage_location)
    return files

def get_new_metric_storage_identifier ():
    existing_files = get_metric_storage_filenames()
    ids = []
    if len(existing_files) == 0:
        metric_storage_id = 1
        return metric_storage_id

    for file in existing_files:
        ids.append(int(file.split(".")[0]))
    ids.sort()
    metric_storage_id = int(ids[-1]) + 1
    return metric_storage_id

def store_metrics (metrics_dict, emotion, algorithm, modifier, n_grams):
    location = ds.metric_storage_location
    metric_id = get_new_metric_storage_identifier()
    metric_list_for_df = []

    for index in range(1,11):
        metric_list_for_df.append([
            metric_id,
            index,
            algorithm,
            modifier,
            n_grams,
            metrics_dict[emotion]["precision"][index - 1],
            metrics_dict[emotion]["recall"][index - 1],
            metrics_dict[emotion]["f1-score"][index - 1],
            metrics_dict[emotion]["support"][index - 1],
            metrics_dict["no_" + emotion]["precision"][index - 1],
            metrics_dict["no_" + emotion]["recall"][index - 1],
            metrics_dict["no_" + emotion]["f1-score"][index - 1],
            metrics_dict["no_" + emotion]["support"][index - 1],
            metrics_dict["accuracy"]["list"][index - 1],
            metrics_dict["macro avg"]["precision"][index - 1],
            metrics_dict["macro avg"]["recall"][index - 1],
            metrics_dict["macro avg"]["f1-score"][index - 1],
            metrics_dict["macro avg"]["support"][index - 1],
            metrics_dict["weighted avg"]["precision"][index - 1],
            metrics_dict["weighted avg"]["recall"][index - 1],
            metrics_dict["weighted avg"]["f1-score"][index - 1],
            metrics_dict["weighted avg"]["support"][index - 1],
        ])
    metric_list_for_df.append([
        metric_id,
        "average",
        algorithm,
        modifier,
        n_grams,
        metrics_dict[emotion]["avg"]["precision"],
        metrics_dict[emotion]["avg"]["recall"],
        metrics_dict[emotion]["avg"]["f1-score"],
        metrics_dict[emotion]["avg"]["support"],
        metrics_dict["no_" + emotion]["avg"]["precision"],
        metrics_dict["no_" + emotion]["avg"]["recall"],
        metrics_dict["no_" + emotion]["avg"]["f1-score"],
        metrics_dict["no_" + emotion]["avg"]["support"],
        metrics_dict["accuracy"]["avg"],
        metrics_dict["macro avg"]["avg"]["precision"],
        metrics_dict["macro avg"]["avg"]["recall"],
        metrics_dict["macro avg"]["avg"]["f1-score"],
        metrics_dict["macro avg"]["avg"]["support"],
        metrics_dict["weighted avg"]["avg"]["precision"],
        metrics_dict["weighted avg"]["avg"]["recall"],
        metrics_dict["weighted avg"]["avg"]["f1-score"],
        metrics_dict["weighted avg"]["avg"]["support"],
    ])

    columns = ["metric_dump_id",
                "fold",
                "algorithm",
                "modifier",
                "n_grams",
                emotion + "_precision",
                emotion + "_recall",
                emotion + "_f1-score",
                emotion + "_support",
                "no_" + emotion + "_precision",
                "no_" + emotion + "_recall",
                "no_" + emotion + "_f1-score",
                "no_" + emotion + "_support",
                "accuracy",
                "macro_avg_precision",
                "macro_avg_recall",
                "macro_avg_f1-score",
                "macro_avg_support",
                "weighted_avg_precision",
                "weighted_avg_recall",
                "weighted_avg_f1-score",
                "weighted_avg_support"]

    metric_df = pd.DataFrame(metric_list_for_df, columns=columns)
    helpers.dataframe_to_csv(metric_df, location + str(metric_id) + ".csv")
    return metric_id
