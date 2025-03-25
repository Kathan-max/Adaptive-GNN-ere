import pandas as pd

data = pd.read_csv('lawTrimmed.csv')
print(data.head(2)['tailEmbed'])