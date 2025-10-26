import kfp 
from kfp import dsl

# <-------------------------------------------- COMPONENTS OF PIPELINE -------------------------------------------------------->
def data_ingestion_op():
    return dsl.ContainerOp(

        name = "Data Ingestion",
        image = "sumit017/cancer-app:latest",
        command = ["python", "src/data_ingestion.py"]
    )


def data_processing_op():
    return dsl.ContainerOp(

        name = "Data Processing",
        image = "sumit017/cancer-app:latest",
        command = ["python", "src/data_processing.py"]
    )


def model_training_op():
    return dsl.ContainerOp(

        name = "Model Training",
        image = "sumit017/cancer-app:latest",
        command = ["python", "src/model_training.py"]
    )

# <--------------------------------------------------------  PIPELINE ------------------------------------------------------------->


@dsl.pipeline(
    name = "Cancer Prediction Pipeline",
    description = "Pipeline for cancer prediction."
)
def cancer_prediction_pipeline():

    data_ingestion = data_ingestion_op()
    data_processing = data_processing_op().after(data_ingestion)
    model_training = model_training_op().after(data_processing)



if __name__ == "__main__":

    kfp.compiler.Compiler().compile(

        cancer_prediction_pipeline, "cancer_prediction_pipeline.yaml"
    )