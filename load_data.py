import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob


column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
                'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
data1 = pd.read_csv('data/UNSW-NB15_1.csv', header=None, names=column_names, low_memory=False)
data2 = pd.read_csv('data/UNSW-NB15_2.csv', header=None, names=column_names, low_memory=False)
data3 = pd.read_csv('data/UNSW-NB15_3.csv', header=None, names=column_names, low_memory=False)
data4 = pd.read_csv('data/UNSW-NB15_4.csv', header=None, names=column_names, low_memory=False)


data = data1
data.fillna('0', inplace=True)

data.drop(columns=['Label'], inplace=True)
data.rename(columns={"attack_cat": "label"}, inplace=True)

le = LabelEncoder()

for col in ['srcip', 'sport', 'dsport', 'dstip', 'proto', 'state', 'service', 'ct_ftp_cmd', 'is_ftp_login',
            'ct_flw_http_mthd', 'label']:
    data[col] = data[col].astype(str)
    data[col] = le.fit_transform(data[col])

data['proto_copy'] = le.fit_transform(data['proto'])

cols = list(data.columns)
cols[cols.index('proto_copy')], cols[cols.index('label')] = cols[cols.index('label')], cols[cols.index('proto_copy')]
df = data[cols]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# print(X.dtypes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


np.save('data/X.npy', X_scaled)
np.save('data/y.npy', y)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.002, random_state=42)
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.002, random_state=42)

_, X_test1, _, y_test1 = train_test_split(X_scaled, y, test_size=0.3, random_state=6)
_, X_test2, _, y_test2 = train_test_split(X_scaled, y, test_size=0.4, random_state=7)
_, X_test3, _, y_test3 = train_test_split(X_scaled, y, test_size=0.5, random_state=8)
_, X_test4, _, y_test4 = train_test_split(X_scaled, y, test_size=0.6, random_state=9)

y_train = y_train.values

y_test1 = y_test1.values
y_test2 = y_test2.values
y_test3 = y_test3.values
y_test4 = y_test4.values

np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)

np.save('data/X_test1.npy', X_test1)
np.save('data/y_test1.npy', y_test1)

np.save('data/X_test2.npy', X_test2)
np.save('data/y_test2.npy', y_test2)

np.save('data/X_test3.npy', X_test3)
np.save('data/y_test3.npy', y_test3)

np.save('data/X_test4.npy', X_test4)
np.save('data/y_test4.npy', y_test4)

print('Data processing successful')
