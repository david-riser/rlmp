import numpy as np
import wandb


class PeriodicEvaluator:
    """ This class can be used to periodically evaluate 
        the performance of our algorithm on the environment.
    """
    def __init__(self, eval_function, update_frequency=1):
        self.eval_function = eval_function
        self.update_frequency = update_frequency


    def evaluate(self, step):

        if step % self.update_frequency == 0:
            scores, actions = self.eval_function()

            # Build a monitoring histogram
            unique_actions = np.unique(actions)
            bins = np.arange(np.min(unique_actions) - .5, np.max(unique_actions) + 1.5)
            histo = np.histogram(actions, bins, density=True)
            
            wandb.log({
                "eval_max_score":np.max(scores), "eval_min_score":np.min(scores),
                "eval_mean_score":np.mean(scores), "eval_std_score":np.std(scores),
                "eval_median_score":np.median(scores),
                "eval_actions":wandb.Histogram(np_histogram=histo)
            })

