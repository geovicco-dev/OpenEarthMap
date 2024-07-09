from src.utils.common import execute_pipeline
from src import DataIngestionPipeline, TrainEvaluatePipeline

# execute_pipeline('Data Download and Preparation', DataIngestionPipeline)
execute_pipeline('Train and Evaluate', TrainEvaluatePipeline)