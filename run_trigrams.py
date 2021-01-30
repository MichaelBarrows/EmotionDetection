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
#       dataset for trigrams
def execute_trigrams ():
    # Generate the baseline results for trigrams
    generate_baseline_results.execute(ds.output_trigrams, "trigrams")
    # Generate the inital experiment results for trigrams
    generate_initial_experimental_results.execute(ds.output_trigrams, "trigrams")
    # Create the next experiments for trigrams
    create_next_experiments.run(ds.output_trigrams, "trigrams")
    # Generate the created experiment results for trigrams
    generate_further_experimental_results.execute(ds.output_trigrams, "trigrams")
    # Sort and store best results in separate file
    results_sorting.import_results(ds.output_trigrams)
    results_sorting.import_best_results_and_sort(ds.output_trigrams)

execute_trigrams()
