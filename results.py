import helpers
import dataset as ds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import_results()
# parameters:
#   folder : string - the folder to be used (negation/non negation)
# returns:
#   new_results_df : DataFrame - dataframe containing the results to be used for
#       creating the graph
# description:
#   This function imports all of the best results files for each of the features
#       evaluated (unigrams; bigrams; trigrams; unigrams and bigrams; unigrams,
#       bigrams & trigrams).The data is then added to two lists, one with less
#       details for creating graphs later, and the other with more details for
#       storage. These lists are then transformed into dataframes, with one
#       stored, and one returned.
def import_results (folder):
    unigram_df = helpers.load_dataset(ds.output_data + "unigrams/" + folder + "/best_result_per_mpt.csv")
    bigram_df = helpers.load_dataset(ds.output_data + "bigrams/" + folder + "/best_result_per_mpt.csv")
    trigram_df = helpers.load_dataset(ds.output_data + "trigrams/" + folder + "/best_result_per_mpt.csv")
    unigram_bigram_df = helpers.load_dataset(ds.output_data + "unigrams_bigrams/" + folder + "/best_result_per_mpt.csv")
    unigram_bigram_trigram_df = helpers.load_dataset(ds.output_data + "unigrams_bigrams_trigrams/" + folder + "/best_result_per_mpt.csv")
    new_results, res_for_storage = [], []
    for idx in range(0, 100, 10):
        unigram = unigram_df[unigram_df.mpt == idx]
        bigram = bigram_df[bigram_df.mpt == idx]
        trigram = trigram_df[trigram_df.mpt == idx]
        unigram_bigram = unigram_bigram_df[unigram_bigram_df.mpt == idx]
        unigram_bigram_trigram = unigram_bigram_trigram_df[unigram_bigram_trigram_df.mpt == idx]
        new_results.append([idx, unigram['weighted_avg_f1-score'].tolist()[0], bigram['weighted_avg_f1-score'].tolist()[0], trigram['weighted_avg_f1-score'].tolist()[0], unigram_bigram['weighted_avg_f1-score'].tolist()[0], unigram_bigram_trigram['weighted_avg_f1-score'].tolist()[0]])
        res_for_storage.append([idx,
                                unigram['weighted_avg_f1-score'].tolist()[0],
                                unigram['algorithm'].tolist()[0],
                                unigram['hyperparameter'].tolist()[0],
                                bigram['weighted_avg_f1-score'].tolist()[0],
                                bigram['algorithm'].tolist()[0],
                                bigram['hyperparameter'].tolist()[0],
                                trigram['weighted_avg_f1-score'].tolist()[0],
                                trigram['algorithm'].tolist()[0],
                                trigram['hyperparameter'].tolist()[0],
                                unigram_bigram['weighted_avg_f1-score'].tolist()[0],
                                unigram_bigram['algorithm'].tolist()[0],
                                unigram_bigram['hyperparameter'].tolist()[0],
                                unigram_bigram_trigram['weighted_avg_f1-score'].tolist()[0],
                                unigram_bigram_trigram['algorithm'].tolist()[0],
                                unigram_bigram_trigram['hyperparameter'].tolist()[0]])
    new_results_df = pd.DataFrame(new_results, columns=["mpt", "Unigrams", "Bigrams", "Trigrams", "Unigrams & Bigrams", "Unigrams, Bigrams and Trigrams"])
    cols = ["mpt", "unigram_f1", "unigram_algorithm", "unigram_hyperparameter", "bigram_f1", "bigram_algorithm", "bigram_hyperparameter", "trigram_f1", "trigram_algorithm", "trigram_hyperparameter", "unigram_bigram_f1", "unigram_bigram_algorithm", "unigram_bigram_hyperparameter", "unigram_bigram_trigram_f1", "unigram_bigram_trigram_algorithm", "unigram_bigram_trigram_hyperparameter"]
    res_for_storage_df = pd.DataFrame(res_for_storage, columns=cols)
    helpers.dataframe_to_csv(res_for_storage_df, ds.output_data + "results/best_results_" + folder + ".csv")
    return new_results_df

# generate_graph()
# parameters:
#   data : DataFrame - the dataframe containing the results data for the line
#       graph
#   filename : string - the filename (and path) for storing the generated graph
#   title : string - the title for the graph
# returns:
#   None
# description:
#   This function sets the index on the dataframe, the style for the graph and
#       plots the graph which is then stored
def generate_graph (data, filename, title):
    data = data.set_index('mpt')
    plt.style.use('fivethirtyeight')
    ax = data.plot.line(figsize=(20,11))
    plt.xlabel("\nMatch Percentage Threshold")
    plt.ylabel("f1-score")
    plt.title(title)
    plt.subplots_adjust(bottom=0.175)
    plt.savefig(filename)
    plt.close()

# process the negation not handled dataset results
results_df = import_results("negation_not_handled")
generate_graph (results_df, ds.output_data + "results/best_results_negation_not_handled.png", "Best results - negation not handled")

# process the negation handled dataset results
results_df = import_results("negation_handled")
generate_graph (results_df, ds.output_data + "results/best_results_negation_handled.png", "Best results - negation handled")
