import pandas as pd
if __name__ == '__main__':

    climate = pd.read_csv("data/climate.csv", encoding="cp1255")
    rain = pd.read_csv("data/rain.csv", encoding="cp1255")

    climate["rain"] = climate["date"].isin(rain["date"].values)
    climate.to_csv("data/full_climate.csv")
