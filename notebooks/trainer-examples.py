from gretel_trainer import trainer

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# Simplest example
model = trainer.Trainer()
model.train(dataset)
df = model.generate()

# Specify underlying model
#model = trainer.Trainer(model_type="GretelLSTM")
#model.train(dataset)
#df = model.generate()

# Update trainer parameters
#model = trainer.Trainer(max_header_clusters=20, max_rows=50000)
#model.train(dataset)
#df = model.generate()

# Specify synthetic model and update config params
#model = trainer.Trainer(model_type="GretelCTGAN", model_kwargs={'epochs': 20})
#model.train(dataset)
#df = model.generate()

# Load and generate data from an existing model
#model = trainer.Trainer()
#model.load()
#df = model.generate(num_records=500)

print(df)