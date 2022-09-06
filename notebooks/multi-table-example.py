from gretel_trainer.relational import MultiTable

#!wget https://gretel-blueprints-pub.s3.amazonaws.com/rdb/ecom_xf.db
mt = MultiTable("sqlite:///ecom_xf.db")
mt.fit()

samples = mt.sample(record_size_ratio=2)

for table_name, df in samples.items():
    df.to_csv(f"synthetic-{table_name}.csv", index=False)
