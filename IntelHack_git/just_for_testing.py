import pandas as pd
import numpy as np
import seaborn as sns
import baseline
from create_grid import make_grid
from sklearn.model_selection import train_test_split
from collections import Counter
from street_converter import net_xy

def generate_df_with_xy(df, data = False):
    """
    :param df: data frame to fill x,y cords with
    :param data: if True, collect street dict pickle from local folder, else, create it now
    :return:
    """

    def use_street_dicts_to_match_df_data():
        """
        Take the street dict calcualted in the function, iterates over the rows and while applying the
        street standirization and adds the correct x,y coords for each row
        :return: xs, ys coords corrspending to each row in the df
        """
        xs = []
        ys = []
        for s in df['street'][index].values:
            if s and type(s) == str and s != np.nan:
                s = s.replace('כיכר', 'רחוב')
                s = s.replace('שביל', 'רחוב')
                s = s.split('/')[0]
                if s in x_street_dict.keys():
                    xs.append(x_street_dict[s])
                    ys.append(y_street_dict[s])
                else:
                    xs.append(0)
                    ys.append(0)
            else:
                xs.append(0)
                ys.append(0)
        return xs, ys
    def get_unique_streets():
        """
        calcs the set of streets after cleanup, if data param then load them from pickle
        :return:
        """
        x_street_dict = {}
        y_street_dict = {}

        if data:
            x_street_dict = load_from_pickle('x_street_dict.pkl')
            y_street_dict = load_from_pickle('y_street_dict.pkl')
            unique_streets = [k for k in x_street_dict.keys() if x_street_dict[k] == 0]
            # the last step is to take in the street we could not find while looking for coords
        else:
            # some standardization for street names
            unique_streets = set(df['street'].unique())
            unique_streets = [s for s in unique_streets if s and s is not None and s is not np.nan and type(s) == str]
            unique_streets = [s.replace("\"", '') for s in unique_streets]
            unique_streets = [s.replace("\'", '') for s in unique_streets]
            unique_streets = [s.replace("-", '') for s in unique_streets]
        return unique_streets ,  x_street_dict ,y_street_dict

    def get_street_to_coords_mapping():
        """
        calls roy nevelas coord getter using the unique streets set
        :return: x_street_dict, y_street_dict
        """
        missed_counter = 0
        for i, s in enumerate(unique_streets):
            print('i {} / {}, street {}'.format(i, len(unique_streets), s))
            s = s.replace('כיכר', 'רחוב')
            s = s.replace('שביל', 'רחוב')
            s = s.replace('שביל', 'רחוב')
            s = s.split('/')[0]
            if s and s is not None and s is not np.nan and type(s) == str:

                tup = net_xy(s + ',עכו')  # here we get the coords from Nevela's call to gov.il
                if tup[0] == 0:
                    # this means we couldnt get the coords
                    print('couldent get {}'.format(tup[0]))
                    missed_counter += 1
                # matches street name to collected coord!
                x_street_dict[s] = tup[0]
                y_street_dict[s] = tup[1]
        print('number of missed xys {}'.format(missed_counter))
        print('number of found xys {} out of {}'.format(len(x_street_dict.keys()), len(unique_streets)))
        input('if you press enter it will save the dicts to pk')
        write_to_pickle('x_street_dict.pkl', x_street_dict)
        write_to_pickle('y_street_dict.pkl', y_street_dict)
        return x_street_dict, y_street_dict

    # get the index of all the rows we dont have x,y's coords in yet - we do not want to overwrite anything
    index = df['x'].index[df['x'].apply(pd.isnull)]
    unique_streets , x_street_dict , y_street_dict = get_unique_streets()
    xs, ys = use_street_dicts_to_match_df_data()
    x_street_dict , y_street_dict = get_street_to_coords_mapping()

    # append the xs and ys to the df rows
    df['x'][index] = xs
    print(len(df['x'].values))
    df['y'][index] = ys
    df.to_pickle('data/df_with_xy_1.pkl')
    print('done!')

def pre_process(df):
    df = df[df['simplified_scenario'] != 'הפסקה']
    cnt = Counter(df['simplified_scenario'].values)
    print('begining number of scenarios {}'.format(len(cnt.keys())))
    legit_scenarios = [k for k in cnt.keys() if cnt[k] > 250]
    print('new number of scenraios {}'.format(len(legit_scenarios)))
    df = df[df['simplified_scenario'].isin(legit_scenarios)]
    df.to_pickle('cleaned_df.pk')
    print('done')

def load_from_pickle(path):
    import pickle
    return pickle.load(open(path, 'rb'))

def write_to_pickle(path, data):
    import pickle
    return pickle.dump(data, open(path, 'wb'))

def do_some_plots(df):
    print(df.columns)
    input()
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    cnt = Counter(df['simplified_scenario'].values)
    scenraios = cnt.keys()
    amount = cnt.values()

    # objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(scenraios))

    plt.bar(y_pos, amount, align='center', alpha=0.5)
    plt.xticks(y_pos, amount)
    plt.ylabel('Usage')
    plt.title('Programming language usage')

    # plt.show()
    # sns.distplot(cnt)
    plt.show()
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
# def remove_break():
#     df = df[df['secen']]

def train_for_grids():
    """
    should load df, get grids from x, y coords. group by main df by grids and train model for the top 'event'
     occuring in that grid.
    :return: event prob for each grid
    """
    # TODO im not sure this is working !
    df = pd.read_pickle('data/cleaned_with_xy.pkl')
    grids = []
    dfs = []
    labs, bounds = make_grid(df['x'].to_numpy(), df['y'].to_numpy(), n_squares=10)
    df['grid'] = ["{},{}".format(a, b) for a, b in zip(labs[0], labs[1])]
    for g in df.groupby('grid'):
        cur_grid = g[0]
        cur_df = g[1]
        grids.append(cur_grid)
        dfs.append(cur_df)
    print('grid creation done')
    results = dict()
    for g, d in zip(grids, dfs):
        d_dict = baseline.run_baseline(d)
        #TODO check if this really give the top event
        top_event = Counter(list(d['simplified_scenario'].values)).most_common(1)[0][0]
        print(top_event)
        # for event in d['simplified_scenario'].unique():
        if top_event in d_dict.keys():
            model, X_train, X_test, y_train, y_test = d_dict[top_event]
            try:
                print('entered train')
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                results[g] = (top_event, pred)
            except:
                print('fail')
    return results












if __name__ == '__main__':
    ret = train_for_grids()
    np.save('dict.npy', ret)

