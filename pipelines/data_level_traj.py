from pipelines import solver_base


class DataTrjPipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(DataTrjPipeline, self).__init__(config, logger)
                #Early stopping 
        self.counter = 0
        self.metric_1 = 1 # TODO
        self.metric_2 = 2 # TODO 