import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv

rain_path = "./data/rain.csv"
csv_path = "./data/acco_semicleaned.csv"
def_col = ('rain_mm', 'event_year', 'event_month', 'event_weekday')
def_type = ('f', 'l', 'l', 'l')
pickle_path = "./data/cleaned_df.pkl"
DATE_LEN = 10
TIME_LEN = 8
MAX_WEEKS = 53


def get_rain_dict(rain_path):
    rainy_days = {}
    with open(rain_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i == 0: continue
            rainy_days[row[0]] = float(row[1])
    return rainy_days


def add_rain_to_df(df, rain_path=rain_path):
    rainy_days = get_rain_dict(rain_path)
    for i in range(1, len(df)):
        if i%5000 == 0:
            print(i)
        df.loc[i, 'rain_mm'] = rainy_days.get(df.loc[i, 'start_event_date'], 0)


def read_clean_df(csv_path=csv_path):
    df = pd.read_csv(csv_path, encoding='cp1255')
    # print('make sure it looks okay and all the columns have the right type\n', df.head())
    # print('keys: ', df.keys())
    # print(df.shape)
    df = df[df['id'].notnull()]
    df[df['start_event_time'].apply(lambda x: type(x) != str)] = np.nan
    df[df['start_event_date'].apply(lambda x: type(x) != str)] = np.nan
    df.dropna(subset=['start_event_time'])
    df.dropna(subset=['start_event_date'])
    # print(df.shape)
    df = df[df['scenario'].notnull()]
    # print(df.shape)
    df.fillna(-999)
    df['end_event_date'].fillna(df['start_event_date'], inplace=True)
    df['end_event_time'].fillna(df['start_event_time'], inplace=True)
    df = process_times(df)
    df = simplify_scenarios(df)
    add_rain_to_df(df)
    print(df.shape)
    print(df.keys())
    return df


def process_times(df: pd.DataFrame) -> pd.DataFrame:
    mydateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y%H:%M:%S")
    start = df[['start_event_date', 'start_event_time']].apply(lambda x: ''.join(x), axis=1).apply(mydateparser)
    end = df[['end_event_date', 'end_event_time']].apply(lambda x: ''.join(x), axis=1).apply(mydateparser)
    df['event_year'] = start.dt.year.astype(int)
    df['event_month'] = start.dt.month.astype(int)
    df['event_day'] = start.dt.day.astype(int)
    df['event_weekday'] = start.dt.weekday_name
    df['event_hour'] = start.dt.hour
    df['event_duration'] = (pd.to_datetime(end) - pd.to_datetime(start)) / pd.Timedelta(1, 'm')
    # df.drop(['start_event_date', 'start_event_time',
    #          'end_event_date', 'end_event_time'], axis=1, inplace=True)
    return df


def list_IoU(l1: list, l2: list) -> float:
    intersection = np.sum([l1[i]==l2[i] for i in range(min(len(l1), len(l2)))])
    union = max(len(l1), len(l2))
    return intersection/union


def simplify_scenarios(df: pd.DataFrame, th: float=.6) -> pd.DataFrame:
    scenario = df['scenario'].to_numpy()
    scenes = np.unique(scenario)
    un = [a.split('-')[0].replace('\\', ' ')
           .replace('/', ' ').replace('שיטור עירוני', '')
           .split()
          for a in np.unique(scenario)]
    scene_dict = {a: a.split('-')[0] for a in scenes}
    D = np.zeros((len(un), len(un)))
    for i, name1 in enumerate(un):
        for j, name2 in enumerate(un[i:]):
            # D[i, j] = len(un[i].intersection(un[j]))/len(un[i].union(un[j]))
            D[i, j] = list_IoU(un[i], un[j])
    D[D < th] = 0
    D[D >= th] = 1
    for i, name in enumerate(scenes):
        # print(i, ':', name)
        inds = np.where(D[i, :] == 1)
        for ind in inds[0][1:]:
            scene_dict[scenes[ind]] = scene_dict[name]
            D[ind, :] = 0
    # for i, a in enumerate(scene_dict):
        # print('ind:', i, 'original:', a, 'new:', scene_dict[a])
    simplified = [scene_dict[a] for a in scenario]
    df['simplified_scenario'] = simplified
    # print(df.keys())
    return df


def get_Xy(df: pd.DataFrame, columns: list=def_col, types: list=def_type):
    if len(types) > len(columns): Exception('More types than columns were given')
    if len(columns) != len(types): types = types + ['l']*(len(columns) - len(types))
    X = np.zeros((df.shape[0], len(columns)))
    for i, c in enumerate(columns):
        if c not in df.keys(): Exception('Column ' + c + ' not in dataframe')
        col = df[c].to_numpy()
        if types[i] == 'l':
            le = LabelEncoder()
            col = le.fit_transform(col)
            print(col)
            X[:, i] = col
        elif types[i] == 'b':
            col[col > 0] = 1
            X[:, i] = col
        elif types[i] == 'f':
            X[:, i] = col.astype(float)
        elif types[i] == 'i':
            X[:, i] = col.astype(int)
        else:
            Exception('Type ' + types[i] + ' not supported')
    y = pd.get_dummies(df['simplified_scenario'])
    # y = LabelEncoder().fit_transform(['simplified_scenario'])
    return X, y


def create_regression_Xy(df: pd.DataFrame, look_back: int=53):
    scenarios = df['simplified_scenario']
    mydateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y")
    week = df[['event_day', 'event_month', 'event_year']].astype(str) \
        .apply(lambda x: '/'.join(x), axis=1).apply(mydateparser).dt.week
    week = week.to_numpy()
    year = df['event_year'].to_numpy()
    numbered_years = year - np.min(year)
    numbered_weeks = [week[i] + MAX_WEEKS*(numbered_years[i]) for i in range(len(week))]
    numbered_weeks = np.array(numbered_weeks) - np.min(numbered_weeks)

    X_dict = {a: [] for a in np.unique(scenarios)}
    y_dict = {a: [] for a in np.unique(scenarios)}
    for event in np.unique(scenarios):
        event_weeks = numbered_weeks[scenarios == event]
        # event_counts = np.array([np.sum(event_weeks == i) for i in np.unique(numbered_weeks)])
        event_counts = np.array([np.sum(event_weeks == i) for i in np.arange(np.max(numbered_weeks))])
        X_dict[event] = np.vstack([event_counts[i: i + look_back]
                                   for i in range(len(event_counts)-look_back-1)])
        y_dict[event] = np.array([event_counts[i] for i in range(look_back + 1, len(event_counts))])
    return X_dict, y_dict


# a = read_clean_df('data/acco_semicleaned.csv')
# X, y = get_Xy(a)
# print(X.shape, y.shape)
# df = pd.read_pickle(pickle_path)
# df.to_pickle(pickle_path)