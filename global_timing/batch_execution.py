"""ML search wrapper for shared independent replay and full validation."""
import os

from .pipeline import Experiment
from .execution import execute_many, replay_many


class BatchedExperiment(Experiment):
    def __init__(self, *args, validation_workers=2, on_completion=None, **kwargs):
        if type(validation_workers) is not int or not 1 <= validation_workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')):
            raise ValueError('Validation workers must fit the Slurm CPU allocation.')
        self.validation_workers = validation_workers
        self.on_completion = on_completion
        super().__init__(*args, **kwargs)

    evaluate_many = execute_many
