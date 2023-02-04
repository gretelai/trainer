import pandas as pd
import gretel_trainer.b2 as b2

df = pd.read_csv("~/Downloads/bikebuying.csv")
bikes = b2.make_dataset(df, datatype="tabular_numeric", name="bikes")
tiny = b2.make_dataset("~/Downloads/tiny.csv", datatype=b2.Datatype.tabular)

repo = b2.GretelDatasetRepo()
iris = repo.get_dataset("iris")

comparison = b2.Comparison(datasets=[bikes, tiny, iris], models=[b2.GretelACTGAN, b2.GretelAmplify])
