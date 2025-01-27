import pandas as pd
ship_data = pd.read_csv("SAM.txt", delimiter= '\s+', index_col=False)

ship_data = ship_data.set_index('type')

ship_data = ship_data.transpose()
ship_dict = ship_data.to_dict(orient='index')
print(ship_dict)