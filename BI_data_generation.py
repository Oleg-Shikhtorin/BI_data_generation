import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm


def save_df_to_csv(name, df):
    df.to_csv(f'{name}.csv', encoding='windows-1251', index=False)


stores = pd.read_csv('stores.csv', parse_dates=[
                     'date_start', 'rent_price_date'])

with open('goods.txt', 'r') as f:
    goods_list = [string.replace('\n', '') for string in f.readlines()]

with open('goods_probabilities.txt', 'r') as f:
    goods_probabilities = [float(string.replace('\n', ''))
                           for string in f.readlines()]

with open('locations.txt', 'r') as f:
    locations = [string.replace('\n', '') for string in f.readlines()]

with open('location_probabilities.txt', 'r') as f:
    location_probabilities = [float(string.replace('\n', ''))
                              for string in f.readlines()]

sell_prices = [
    range(80000, 130001, 5000),
    range(5000, 130001, 1000),
    range(120, 15001, 500),
    range(200, 7001, 100),
    range(250, 10001, 100),
    range(100, 501, 50),
    range(100, 4501, 10),
    range(25000, 120001, 5000),
    range(15000, 200001, 5000),
    range(500, 5001, 500),
    range(500, 5001, 500),
    range(500, 5001, 500),
    range(500, 10001, 1000),
    range(2000, 20001, 1000),
    range(10000, 50001, 5000),
    range(2000, 20001, 1000),
    range(500, 5001, 1000),
    range(500, 10001, 1000),
    range(500, 10001, 1000),
    range(500, 10001, 500),
    range(500, 10001, 500),
    range(500, 10001, 200),
]


def create_product_table(goods_list, goods_probabilities, sell_prices):
    name = 'products'
    print(f'started creation dataframe {name}...')

    cost_prices_coef = list(np.arange(.76, .99, .005))

    prod_start = '2015-01-10'
    prod_end = '2015-09-01'
    prod_date_range = pd.date_range(start=prod_start, end=prod_end)

    df = pd.DataFrame(
        {'goods': np.random.choice(
            goods_list,
            p=goods_probabilities,
            size=1000)}
    )

    for row in range(len(df)):
        idx = goods_list.index(df.loc[row, 'goods'])
        price = np.random.choice(list(sell_prices[idx]))
        df.loc[row, 'sell_price'] = price

    df['category'] = df['goods'].apply(lambda x: x.split('|')[0])
    df['subcategory'] = df['goods'].apply(lambda x: x.split('|')[1])

    df['coef'] = np.random.choice(cost_prices_coef, size=1000)
    df['cost_price'] = round((df['sell_price'] * df['coef']), 2)

    none = df[df['subcategory'] == 'null'].index
    df.loc[none, 'subcategory'] = None

    df['price_date'] = pd.Series(
        np.random.choice(prod_date_range, size=1000)
    )
    df = df.drop(columns=['goods', 'coef']).sort_values(
        'price_date').reset_index(drop=True)
    df['id'] = range(1, len(df)+1)
    df = df[['id', 'sell_price', 'category',
             'subcategory', 'price_date', 'cost_price']]

    print(f'created dataframe {name} with shape {df.shape}', end='\n')

    return (name, df)


name, products = create_product_table(
    goods_list, goods_probabilities, sell_prices)
save_df_to_csv(name, products)

# user registration dates are between 2015-01-01 and current day
# total number of users a little bit greater then 500 000
n = int(500000 * np.random.choice(np.arange(1, 1.16, .0001)))

birth_point_0 = '1956-01-01'
birth_point_1 = '1990-01-01'
birth_point_2 = '2010-01-01'

reg_start = '2015-01-01'
reg_end = datetime.datetime.now().date()

# two date ranges (i'll do them different sizes)
birth_range_older = pd.date_range(
    birth_point_0, birth_point_1)  # 0.3 * total users
birth_range_younger = pd.date_range(
    birth_point_1, birth_point_2)  # 0.7 * total users
older = np.random.choice(birth_range_older, size=int(n*.3))
younger = np.random.choice(birth_range_younger, size=n - int(n*.3))

# vector of registration dates
reg_range = pd.date_range(reg_start, reg_end)


def create_table_users(birth, reg, locations, location_probabilities, n):
    name = 'users'
    print(f'started creation dataframe {name}...')

    if len(birth) == 2:
        older, younger = birth[0], birth[1]
        df_dict = {
            'location': np.random.choice(locations, p=location_probabilities, size=n),
            'birth_date': np.concatenate((older, younger)),
            'reg_date': np.random.choice(reg, size=n),
            'gender_int': np.random.binomial(1, .57, size=n)
        }
    else:
        df_dict = {
            'location': np.random.choice(locations, p=location_probabilities, size=n),
            'birth_date': np.random.choice(birth, size=n),
            'reg_date': np.random.choice(reg, size=n),
            'gender_int': np.random.binomial(1, .57, size=n)
        }

    df = pd.DataFrame(df_dict).sort_values(['reg_date'])
    male = df[df['gender_int'] == 1].index
    df['gender'] = 'female'
    df.loc[male, 'gender'] = 'male'

    df['country'] = df['location'].apply(lambda x: x.split('|')[0])
    df['city'] = df['location'].apply(lambda x: x.split('|')[1])

    df['id'] = range(1, len(df)+1)
    df = df[['id', 'gender', 'birth_date', 'country',
             'city', 'reg_date']].reset_index(drop=True)

    print(f'created dataframe {name} with shape {df.shape}', end='\n')

    return (name, df)


name, users = create_table_users(
    [older, younger], reg_range, locations, location_probabilities, n)
save_df_to_csv(name, users)

# users to order
n = int(200000 * np.random.choice(np.arange(0.97, 1.07, .0001)))

states_list = ['PAID', 'CANCELLED', 'CREATED']
states_probabilities = [.7, .1, .2]

channel_list = ['mobile', 'web', 'offline']
channel_probabilities = [.367, .215, .418]

delivery_list = ['post', 'pickup', 'carrier']
delivery_probabilities = [.123, .717, .16]


def create_table_orders(users_df, stores_df, states_list, states_probabilities, channel_list, channel_probabilities,
                        delivery_list, delivery_probabilities, n):
    name = 'orders'
    print(f'started creation dataframe {name}...')

    today = datetime.datetime.now().date()

    df = users_df[['id', 'reg_date']].sample(n).sample(n*2, replace=True)
    df['order_date'] = df['reg_date'].apply(
        lambda x: np.random.choice(pd.date_range(x, today)))

    df['state'] = np.random.choice(
        states_list, p=states_probabilities, size=len(df))
    df['channel'] = np.random.choice(
        channel_list, p=channel_probabilities, size=len(df))
    df['delivery_type'] = np.random.choice(
        delivery_list, p=delivery_probabilities, size=len(df))
    df['store_id'] = np.random.choice(
        list(range(1, stores_df['id'].nunique()+1)), size=len(df))

    df = df.sort_values(['order_date']).reset_index(drop=True)

    pickup = df[df['channel'] == 'offline'].index
    df.loc[pickup, 'delivery_type'] = 'pickup'

    store_0 = df[(df['channel'] != 'offline') & (
        df['delivery_type'] != 'pickup')].index
    df.loc[store_0, 'store_id'] = 0

    df.rename(columns={'id': 'user_id'}, inplace=True)
    df['id'] = range(1, len(df)+1)
    df['basket_id'] = range(1, len(df)+1)
    df = df[['id', 'basket_id', 'user_id', 'order_date',
             'state', 'channel', 'delivery_type', 'store_id']]

    print(f'created dataframe {name} with shape {df.shape}', end='\n')

    return (name, df)


name, orders = create_table_orders(users, stores, states_list, states_probabilities, channel_list, channel_probabilities,
                                   delivery_list, delivery_probabilities, n)
save_df_to_csv(name, orders)

counter = range(1, 11)
counter_probabilities = [
    .601,
    .165,
    .074,
    .05,
    .03,
    .02,
    .02,
    .02,
    .01,
    .01
]


def create_table_baskets(orders_df, products_df, counter, counter_probabilities):
    name = 'baskets'
    print(f'started creation dataframe {name}...')

    basket_list = orders_df['basket_id'].unique().tolist()
    product_ser = pd.Series(products_df['id'].unique())

    baskets = []
    products = []
    cnts = []

    for basket in tqdm(basket_list):
        for _ in range(0, np.random.choice(counter, p=counter_probabilities)):
            baskets.append(basket)
            products.append(product_ser.sample(1).values[0])
            cnts.append(np.random.choice(counter, p=counter_probabilities))
    df = pd.DataFrame(
        {
            'id': baskets,
            'product_id': products,
            'cnt': cnts
        }
    )
    print(f'created dataframe {name} with shape {df.shape}', end='\n')

    return (name, df)


name, baskets = create_table_baskets(
    orders, products, counter, counter_probabilities)
save_df_to_csv(name, baskets)


def add_random_time(date):
    hour = np.random.choice(range(24))
    minute = np.random.choice(range(60))
    second = np.random.choice(range(60))
    return datetime.datetime.combine(date, datetime.time(hour, minute, second))


# this part is about registration
reg_source = ['org', 'ad', 'social_network']
reg_probabilities = [.351, .405, .244]


def reg_events(users_df, reg_source, reg_probabilities):
    print('creating registration events...')
    df = users_df[['id', 'reg_date']].copy()

    # add random reg time to date
    df.loc[:, 'dt'] = df.loc[:, 'reg_date'].apply(add_random_time)

    # add event type
    df.loc[:, 'type'] = 'reg'
    df = df[['id', 'dt', 'type']]

    # add source
    df.loc[:, 'source'] = np.random.choice(
        reg_source, p=reg_probabilities, size=len(df))

    print('registration events created', end='\n')

    return df


def first_enter(dt):
    seconds = np.random.choice(range(1, 4)).item()
    return dt + datetime.timedelta(seconds=seconds)

# same users, first enter and escape then


def first_enter_and_escape(reg_events_df):
    print('creating first enter and escaping...')
    df = reg_events_df[['id', 'dt']].copy()
    df.loc[:, 'enter_dt'] = df.loc[:, 'dt'].apply(first_enter)
    df.loc[:, 'type'] = 'enter'
    df.loc[:, 'page'] = 1

    a = []  # user's id
    b = []  # event_types
    c = []  # pages
    d = []  # time

    pbar = tqdm(total=(len(df['id'])))
    for user, dt in zip(df['id'].tolist(), df['enter_dt'].tolist()):
        # 98% users visit more then 1 page per registration session
        if np.random.binomial(1, .98) == 1:
            page = None
            time = dt
            for _ in range(np.random.choice(range(1, 4), p=[.8, .15, .05])):
                a.append(user)
                b.append('enter')

                page = np.random.choice(range(2, 13))
                c.append(page)

                seconds = np.random.choice(range(4, 601)).item()
                time = time + datetime.timedelta(seconds=seconds)
                d.append(time)

        # escaping
        page = np.random.choice(range(2, 13))
        seconds = np.random.choice(range(4, 601)).item()
        time = time + datetime.timedelta(seconds=seconds)

        a.append(user)
        b.append('close')
        c.append(page)
        d.append(time)
        pbar.update(1)
    pbar.close()

    first_enter_and_escape_df = pd.DataFrame(
        {'id': a, 'dt': d, 'type': b, 'page': c})
    
    print('first enter and escaping created', end='\n')

    return first_enter_and_escape_df


def reg_first_enter_and_escape_events(reg_events_df, first_enter_and_escape_df):
    df = pd.concat(
        [
            reg_events_df,
            first_enter_and_escape_df
        ]
    ).sort_values(['id', 'dt']).reset_index(drop=True)
    return df


reg_events_df = reg_events(users, reg_source, reg_probabilities)
first_enter_and_escape_df = first_enter_and_escape(reg_events_df)
reg_first_enter_and_escape_events_df = reg_first_enter_and_escape_events(
    reg_events_df, first_enter_and_escape_df)

# generating site usage history


def site_usage_events(reg_first_enter_and_escape_events_df):
    print('creating site usage events...')
    # last event time of each user
    df = reg_first_enter_and_escape_events_df.groupby('id').tail(1)

    # 87% of them will visit site more then ones
    df = df[['id', 'dt']].sample(int(len(df)*.87))

    a = []  # user's id
    b = []  # event_types
    c = []  # pages
    d = []  # time

    pbar = tqdm(total=(len(df)))
    for user_, dt_ in zip(df['id'].tolist(), df['dt'].tolist()):  # user and last datetime
        now = datetime.datetime.now()
        visit_range = pd.date_range(start=dt_.date(), end=now)
        visit_cnt = np.random.choice(
            range(1, 9), p=[.4, .1, .1, .15, .125, .075, .025, .025])
        try:
            visit_days = np.random.choice(visit_range, size=visit_cnt)
        except:
            visit_days = visit_range
        visit_datetimes = [
            add_random_time(
                datetime.datetime.utcfromtimestamp(_.tolist()/1e9)
            ) for _ in visit_days
        ]
        visit_user_ids = [user_] * len(visit_datetimes)

        # generating sessions
        for user, dt in zip(visit_user_ids, visit_datetimes):
            page = None
            time = dt

            # browsing into session
            for _ in range(np.random.choice(range(1, 4), p=[.8, .15, .05])):
                a.append(user)
                b.append('enter')

                page = np.random.choice(range(2, 13))
                c.append(page)

                seconds = np.random.choice(range(4, 601)).item()
                time = time + datetime.timedelta(seconds=seconds)
                d.append(time)

            # session escaping
            seconds = np.random.choice(range(4, 601)).item()
            time = time + datetime.timedelta(seconds=seconds)

            a.append(user)
            b.append('close')
            c.append(page)
            d.append(time)

        pbar.update(1)
    pbar.close()
    site_usage_events_df = pd.DataFrame(
        {'id': a, 'dt': d, 'type': b, 'page': c})

    
    print('site usage events created', end = '\n')

    return site_usage_events_df


site_usage_events_df = site_usage_events(reg_first_enter_and_escape_events_df)

# generating orders sessions data


def order_events(orders_df):
    print('creating order events...')
    buyers = orders_df['user_id'].tolist()

    # add some random time to our dates
    buyers_dt = [add_random_time(_) for _ in orders_df['order_date'].tolist()]

    df_orders = pd.DataFrame(
        {
            'id': buyers,
            'dt': buyers_dt
        }
    )
    df_orders['type'] = 'order'

    a = []  # user's id
    b = []  # event_types
    c = []  # pages
    d = []  # time

    pbar = tqdm(total=(len(buyers)))
    for user, dt in zip(buyers, buyers_dt):
        page = None
        time = dt
        # 1-3 events ...
        for _ in range(np.random.choice(range(1, 4), p=[.8, .15, .05])):
            a.append(user)
            b.append('enter')

            page = np.random.choice(range(2, 13))
            c.append(page)

            seconds = np.random.choice(range(4, 601)).item()
            # ... before order
            time = time - datetime.timedelta(seconds=seconds)
            d.append(time)

        page = np.random.choice(range(2, 13))
        seconds = np.random.choice(range(4, 601)).item()
        # escaping after purchase
        time = dt + datetime.timedelta(seconds=seconds)

        a.append(user)
        b.append('close')
        c.append(page)
        d.append(time)
        pbar.update(1)
    pbar.close()

    df_preorder_events = pd.DataFrame({'id': a, 'dt': d, 'type': b, 'page': c})
    order_events = pd.concat([df_orders, df_preorder_events]).sort_values('dt')
    
    print('order events created', end = '\n')

    return order_events


order_events = order_events(orders)

# total events


def create_table_events(reg_first_enter_and_escape_events_df, site_usage_events_df, order_events):
    name = 'events'
    print(f'started creation dataframe {name}...')
    df = pd.concat(
        [
            reg_first_enter_and_escape_events_df,
            site_usage_events_df,
            order_events
        ]
    ).sort_values('dt').reset_index(drop=True)
    df.rename(columns={'id': 'user_id', 'type': 'event_type',
              'source': 'reg_source'}, inplace=True)
    df['id'] = range(1, len(df)+1)
    df = df[['id', 'user_id', 'dt', 'event_type', 'reg_source', 'page']]

    print(f'created dataframe {name} with shape {df.shape}', end='\n')

    return (name, df)


name, events = create_table_events(
    reg_first_enter_and_escape_events_df, site_usage_events_df, order_events)
save_df_to_csv(name, events)
print('Done')
