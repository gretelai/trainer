from gretel_trainer import trainer

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Simplest example
model = trainer.Trainer()
model.train(dataset)
print(model.generate())

# Or, load and generate data from an existing model

#model = trainer.Trainer.load()
#model.generate(num_records=70)
