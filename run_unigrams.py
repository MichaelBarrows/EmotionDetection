import generate_baseline_results
import generate_initial_experimental_results
import create_next_experiments
import generate_further_experimental_results
import results_sorting
import dataset as ds

# execute_negated_dataset()
# parameters:
#   None
# returns:
#   None
# description:
#   This function calls and executes all code linked to the negation not handled
#       dataset for unigrams
def execute_unigrams ():
    # Generate the baseline results for unigrams
    generate_baseline_results.execute(ds.output_unigrams, "unigrams")
    # Generate the inital experiment results for unigrams
    generate_initial_experimental_results.execute(ds.output_unigrams, "unigrams")
    # Create the next experiments for unigrams
    create_next_experiments.run(ds.output_unigrams, "unigrams")
    # Generate the created experiment results for unigrams
    generate_further_experimental_results.execute(ds.output_unigrams, "unigrams")
    # Sort and store best results in separate file
    results_sorting.import_results(ds.output_unigrams)
    results_sorting.import_best_results_and_sort(ds.output_unigrams)


execute_unigrams()
