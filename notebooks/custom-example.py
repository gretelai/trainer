from gretel_trainer import trainer
from gretel_trainer.models import GretelLSTM, GretelCTGAN


dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Specify underlying model and config options.
# configs can be either a string, dict, or path
model_type = GretelCTGAN(
    config="synthetics/high-dimensionality", 
    max_header_clusters=100, 
    max_rows=50000
)

# Optionally update model parameters from a base config
model_type.update_params({"epochs": 500})

model = trainer.Trainer(model_type=model_type)
model.train(dataset)
print(model.generate())

# Or, load and generate data from an existing model
# model = trainer.Trainer.load()
# print(model.generate(num_records=70))
