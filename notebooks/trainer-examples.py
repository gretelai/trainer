from gretel_trainer import trainer

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Simplest example
model = trainer.Trainer()
model.train(dataset)
print(model.generate())

# Or, Specify underlying model and config options.
# Valid options are "GretelLSTM", "GretelCTGAN"

#model = trainer.Trainer(
#    model_type="GretelCTGAN",
#    max_header_clusters=100,
#    max_rows=50000,
#    model_params={"epochs": 600, "num_records": 5500},
#)
#model.train(dataset)
#model.generate()

# Or, load and generate data from an existing model

#model = trainer.Trainer.load()
#model.generate(num_records=70)
