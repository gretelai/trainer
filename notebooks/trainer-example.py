from gretel_trainer import trainer

dataset_path = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

model = trainer.Trainer(model_type="GretelLSTM")
model.train(dataset_path)
print(model.generate())

