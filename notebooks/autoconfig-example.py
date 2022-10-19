from gretel_trainer import trainer


dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"


model = trainer.Trainer()
model.train(dataset)
print(model.generate())
