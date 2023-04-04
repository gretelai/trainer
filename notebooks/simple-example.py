from gretel_client import configure_session
from gretel_trainer import Trainer

# Configure Gretel credentials
configure_session(api_key="prompt", cache="yes", validate=True)

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Simplest example
model = Trainer()
model.train(dataset)
print(model.generate())

# Or, load and generate data from an existing model

#model = Trainer.load()
#model.generate(num_records=70)
