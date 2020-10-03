import requests


def net_xy(street):
    """
    A function that returns the X,Y of the given street
    :param street: a street name in hebrew in format street,city
    :return: a tuple of X,Y cords
    """

    # api-endpoint
    URL = "https://ags.govmap.gov.il/Search/FreeSearch"
    # headers
    headers = {"Content-Type": "application/json", "charset": "utf-8"}
    # location given here
    try:
        p = "{\"keyword\": \"" + street + "\",\"LstResult\": null}"
        PARAMS = p.encode("utf-8")

        # sending get request and saving the response as response object
        r = requests.post(url=URL, data=PARAMS, headers=headers)

        # extracting data in json format
        data = r.json()

        # extracting latitude, longitude and formatted address
        # of the first matching location

        X = data['data']['Result'][0]['X']
        Y = data['data']['Result'][0]['Y']
    except Exception as e:
        print(e)
        # print('exception ddamammnnnnn')
        print(street)
        return 0,0
    return X,Y


def net_xy_to_cords(x,y):
    default_x = 35.0743
    default_y = 32.93016

    if x > 100 and y > 100:
        URL = "https://epsg.io/trans?x=" + str(x) + "&y=" + str(y) + "&s_srs=2039&t_srs=4326"
        r = requests.get(url=URL)
        data = r.json()

        try:
            x = data['x']
            y = data['y']
        except:
            x = default_x
            y = default_y

    elif x < 30 and y < 30:
        x = default_x
        y = default_y
    return x, y
