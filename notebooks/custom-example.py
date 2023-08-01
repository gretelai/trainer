from gretel_client import configure_session
from gretel_trainer import Trainer
from gretel_trainer.models import GretelACTGAN, GretelLSTM

# Configure Gretel credentials
configure_session(api_key="prompt", cache="yes", validate=True)

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Specify underlying model and config options.
# configs can be either a string, dict, or path
model_type = GretelACTGAN(
    config="synthetics/tabular-actgan",
    max_rows=50000
)

# Optionally update model parameters from a base config
model_type.update_params({"epochs": 500})

model = Trainer(model_type=model_type)
model.train(dataset)
print(model.generate())

# Or, load and generate data from an existing model
# model = trainer.Trainer.load()
# print(model.generate(num_records=70))
