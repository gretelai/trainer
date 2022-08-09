import pandas as pd

from gretel_trainer import trainer
from gretel_trainer.models import GretelLSTM, GretelCTGAN

DATASET_PATH = 'https://gretel-public-website.s3.amazonaws.com/datasets/mitre-synthea-health.csv'
MODEL_TYPE = [GretelLSTM(), GretelCTGAN()][1]

# Create dataset to autocomplete values for
seed_df = pd.DataFrame(data=[
    ["black", "african", "F"],
    ["black", "african", "F"],
    ["black", "african", "F"],
    ["black", "african", "F"],
    ["asian", "chinese", "F"],
    ["asian", "chinese", "F"],
    ["asian", "chinese", "F"],
    ["asian", "chinese", "F"],
    ["asian", "chinese", "F"]
], columns=["RACE", "ETHNICITY", "GENDER"])


# Train a model and conditionally generate data
seed_fields = seed_df.columns.values.tolist()
model = trainer.Trainer(model_type=MODEL_TYPE)
model.train(DATASET_PATH, seed_fields=seed_fields)
print(model.generate(seed_df=seed_df))

# Load a existing model and conditionally generate data
#model = trainer.Trainer.load()
#print(model.generate(seed_df=seed_df))