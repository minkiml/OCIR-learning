from pipelines import solver_base


class RL_Trj_pipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(RL_Trj_pipeline, self).__init__(config, logger)