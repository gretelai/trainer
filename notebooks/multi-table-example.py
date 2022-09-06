from gretel_trainer.relational import MultiTable

#!wget https://gretel-blueprints-pub.s3.amazonaws.com/rdb/ecom_xf.db
mt = MultiTable("sqlite:///ecom_xf.db")
synthetic_tables = mt.synthesize_tables(record_size_ratio=2)

for table_name, df in synthetic_tables.items():
    out_path = f"synthetic-{table_name}.csv"
    print(f"Writing to {out_path}")
    df.to_csv(out_path, index=False)
