import pandas as pd
import mne_bids
bids_path = mne_bids.BIDSPath(
 subject = '01',
 session = '0',
 task = '0',
 datatype = "meg",
 root = 'D:\\NTHU\\meta-brainmagick2\\brainmagick\\data\\gwilliams2022\\download')
raw = mne_bids.read_raw_bids(bids_path)
raw.load_data()._data # channels X times
df = raw.annotations.to_data_frame()
df.to_csv('out.csv')