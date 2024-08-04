import mne
import mne_bids
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from tqdm import trange
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
import os

#my self
from tabulate import tabulate
from pprint import pprint

matplotlib.use("Agg")


mne.set_log_level(False)

class BrainwaveAnalyzer:

    def __init__(self, raw, infostr=None):
        self.raw = raw
        self.is_sound_table = None
        self.infostr = infostr

class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
        print(data)
        

    assert data.exists()

    # bids = data / "bids_anonym"
    bids = data     # here set PATH.bids

def createIsSoundTable(raw):
    """
    create table with is_sound column from metadata
    """
    print("createIsSoundTable is running")
    
    meta = list()
    for annot in raw.annotations:

        d = eval(annot.pop("description"))
        # d should be dict_keys containing 
        # ['story', 'story_uid', 'sound_id', 'kind', 'start', 'sound', 'phoneme', 'sequence_id', 'condition','word_index', 'speech_rate', 'voice', 'pronounced']
        
        for k, v in annot.items():  # .items()=key-value pair, k=key, v=value
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    if "intercept" not in meta.columns:
        meta["intercept"] = 1.0
    #meta.to_csv("metadata_in_raw.csv")
    print(tabulate(meta[:30], headers='keys', tablefmt='github'))


    # create is_sound table
    time_step = 0
    rows = []
    # Iterate through df1
    for index, row in meta.iterrows():
        if row['kind'] == 'word':
            # Add row with is_sound = False
            rows.append({'is_sound': False, 'onset': time_step, 'duration': row['onset'] - time_step})
            # Add row with is_sound = True
            rows.append({'is_sound': True, 'onset': row['onset'], 'duration': row['duration']})
            # Update i
            time_step = row['onset'] + row['duration']

    # Create df2
    is_sound_table = pd.DataFrame(rows)
    if(not os.path.exists("is_sound_table.csv")):   
        is_sound_table.to_csv("is_sound_table.csv")

    return

    

def addIsWordCol(raw):
    """
    add a column is_word to meta data
    """
    
    meta = list()
    for annot in raw.annotations:

        d = eval(annot.pop("description"))
        # d should be dict_keys containing 
        # ['story', 'story_uid', 'sound_id', 'kind', 'start', 'sound', 'phoneme', 'sequence_id', 'condition','word_index', 'speech_rate', 'voice', 'pronounced']
        
        for k, v in annot.items():  # .items()=key-value pair, k=key, v=value
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)

    if "intercept" not in meta.columns:
        meta["intercept"] = 1.0


def preprocess_lables_of_metadata(raw):
    """
    first half of orignal segment function
    """
    # preproc annotations
    meta = list()

    # purpose of the below for-loop
    # 1. extract raw.annotations.description into d
    # 2. append all other items in raw.annotations into d
    # make a df from the information
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        # d should be dict_keys containing 
        # ['story', 'story_uid', 'sound_id', 'kind', 'start', 'sound', 'phoneme', 'sequence_id', 'condition','word_index', 'speech_rate', 'voice', 'pronounced']
        
        for k, v in annot.items():  # .items()=key-value pair, k=key, v=value
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')    # extract all rows with kind=="phoneme"
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")   #@ph 即the var ph
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2

    return meta

def segment_epochs(raw, meta):
    """
    original second half of function `segment`
    """
    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)
    # raw.info["sfreq"] = sampling frequency
    # meta.onset * raw.info["sfreq"] = convert conset time to sample index
    # np.ones((len(meta), 2) = create array of same no, of rows as len(meta) and 2 cols
    # 
    #     Summary
    # The resulting events array has the following structure:

    # Each row corresponds to an event.
    # The first column contains the sample indices of the onsets.
    # The second and third columns are filled with ones.

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )


    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs

def segment(raw):
    # preproc annotations
    meta = list()

    # purpose of the below for-loop
    # 1. extract raw.annotations.description into d
    # 2. append all other items in raw.annotations into d
    # make a df from the information
    for annot in raw.annotations:

        d = eval(annot.pop("description"))
        # d should be dict_keys containing 
        # ['story', 'story_uid', 'sound_id', 'kind', 'start', 'sound', 'phoneme', 'sequence_id', 'condition','word_index', 'speech_rate', 'voice', 'pronounced']
        
        for k, v in annot.items():  # .items()=key-value pair, k=key, v=value
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')    # extract all rows with kind=="phoneme"
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")   #@ph 即the var ph
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2

    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)
    # raw.info["sfreq"] = sampling frequency
    # meta.onset * raw.info["sfreq"] = convert conset time to sample index
    # np.ones((len(meta), 2) = create array of same no, of rows as len(meta) and 2 cols
    # 
    #     Summary
    # The resulting events array has the following structure:

    # Each row corresponds to an event.
    # The first column contains the sample indices of the onsets.
    # The second and third columns are filled with ones.

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )

    # remarks
    # tmin, tmax: the time window around each event
    # decim: down sampling factor - e.g. decim =10,which mean down sample by taking 1 sample every 10 sample
    # “baseline” : The time interval to consider as “baseline” when applying baseline correction
    # preload = true: load all data in object creation

    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs

def decod_binary(X, y, meta, times):
    """
    y is expected to be binary with value 0 or 1
    """

def decod(X, y, meta, times):
    assert len(X) == len(y) == len(meta)
    print(f"in decod, len(x)=len(y)=len(meta)={len(X)}")
    # print("--------------------")
    # print("before reset index, first 5 rows of meta:")
    # print(tabulate(meta[:5], headers='keys', tablefmt='github'))
    meta = meta.reset_index()
    # print("--------------------")
    # print("after reset index, first 5 rows of meta:")
    # print(tabulate(meta[:5], headers='keys', tablefmt='github'))


    y = scale(y[:, None])[:, 0]
    # explain
    # y[:, None] transform y from 1D (1*n) arr to 2D (n*1) arr 
    # scale 完後用[:, 0] 轉返做1D arr
    if len(set(y[:1000])) > 2:      # len(set(y[:1000])) > 2 should be a simple way to check if y is binary
        print("in decode, y is not binary (len(set(y[:1000])) > 2), apply binary conversion")
        y = y > np.nanmedian(y)
        # if y > y's median gives True, y<= y's median gives False

    # define data
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, nchans, ntimes = X.shape
    preds = np.zeros((n, ntimes))
    # notes
    # 2nd arg of cross_val_predict is expecting X in shape n_sample, n_feature
    for t in trange(ntimes):
        preds[:, t] = cross_val_predict(
            model, X[:, :, t], y, cv=cv, method="predict_proba"
        )[:, 1]

    # score
    out = list()
    for label, m in meta.groupby("label"):
        print(f"processing label: {label}")
        Rs = correlate(y[m.index, None], preds[m.index])
        # explain
        # label should be 對sample 切 (每個label's group 都有齊81個time point)
        # m is subset of meta in for the current group
        # m.index is the index of the subset
        # y[m.index] select y correspond to these index
        # y[m.index, None] is reshaping y[m.index] to 2D arr
        # see part 1,2 in mytest.py for explanation
        for t, r in zip(times, Rs):
            out.append(dict(score=r, time=t, label=label, n=len(m.index)))
    return pd.DataFrame(out)


def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)


def plot(result):
    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="score", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig





def _get_epochs(subject, until_sesion, until_task, add_is_sound_only_flag=False, to_add_is_sound = None):
    print("_get_epochs is running")
    all_epochs = list()
    for session in range(until_sesion):
        for task in range(until_task):
            print(".", end="")
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=PATHS.bids,
            )
            try:
                raw:mne.io.Raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                print("missing", subject, session, task)
                continue
            raw = raw.pick_types(
                meg=True, misc=False, eeg=False, eog=False, ecg=False
            )

            raw.load_data().filter(0.5, 30.0, n_jobs=1) # n_jobs = Number of jobs to run in parallel.
            if(add_is_sound_only_flag):
                createIsSoundTable(raw)
                print("create is_sound table done. program exit")
                exit(0)
            else:
                epochs = segment(raw)
                epochs.metadata["half"] = np.round(
                    np.linspace(0, 1.0, len(epochs))
                ).astype(int)
                # create a half 0 half 1 column
                epochs.metadata["task"] = task
                epochs.metadata["session"] = session

                all_epochs.append(epochs)


                print("first 5 rows of epochs.metadata")
                print(tabulate(epochs.metadata[:20], headers='keys', tablefmt='github'))
                # add code to check if epochs_metadata.csv exist
                if(not os.path.exists("epochs_metadata.csv")):   
                    epochs.metadata.to_csv("epochs_metadata.csv")

            
    print("breakpoint 1") 
    if not len(all_epochs):
        return
    epochs = mne.concatenate_epochs(all_epochs)
    m = epochs.metadata
    label = (
        "t"
        + m.task.astype(str)
        + "_s"
        + m.session.astype(str)
        + "_h"
        + m.half.astype(str)
    )
    epochs.metadata["label"] = label
    return epochs


def _decod_one_subject(subject, until_session, until_task, add_is_sound_only_flag=False):
    print("_decod_one_subject is running")
    epochs = _get_epochs(subject, until_session, until_task, add_is_sound_only_flag)
    if epochs is None:
        return
    # words
    words = epochs["is_word"]
    print("type of words", type(words))
    # type of words: <class 'mne.epochs.EpochsArray'>
    print("first 5 rows of words.metadata")
    print(tabulate(words.metadata[:5], headers='keys', tablefmt='github'))
    evo = words.average()   # average of is_word ==true
    #  computes the average evoked response for the selected epochs
    # retur: mne.Evoked
    print("type of evo")
    print(type(evo))
    fig_evo = evo.plot(spatial_colors=True, show=False)
    print("breakpoint 2") 
    X = words.get_data() * 1e13     # * 1e13 looks like for transform to ft
    y = words.metadata["wordfreq"].values
    # words.get_data() is expected as numpy 3D array
    # add code to print first few rows in a readbile way
    print("type of y:", type(y))
    # y is expected as numpy 1D array
    # print shape of y
    print("shape of y:", y.shape)
    # print first 5 rows of y
    print("first 5 rows of y")
    pprint(y[:5])
    print("break point")
    print("type of X:", type(X))
    
    print("shpae of X:", X.shape)
    print("break point")
    print("first 5 rows of X")
    pprint(X[:5])


    results = decod(X, y, words.metadata, words.times)
    results["subject"] = subject
    results["contrast"] = "wordfreq"

    fig_decod = plot(results)
    print("program exit before decode phoneme")
    exit(0)

    # Phonemes
    phonemes = epochs["not is_word"]
    evo = phonemes.average()
    fig_evo_ph = evo.plot(spatial_colors=True, show=False)

    X = phonemes.get_data() * 1e13
    y = phonemes.metadata["voiced"].values

    

    results_ph = decod(X, y, phonemes.metadata, phonemes.times)
    results_ph["subject"] = subject
    results_ph["contrast"] = "voiced"
    fig_decod_ph = plot(results_ph)

    return fig_evo, fig_decod, results, fig_evo_ph, fig_decod_ph, results_ph



# ------   global code   ------  #

UNTIL_SUBJECT = 1
UNTIL_SESSION = 1
UNTIL_TASK = 1
ADD_IS_SOUND_ONLY_FLAG = False

ph_info = pd.read_csv("phoneme_info.csv")
subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values

if __name__ == "__main__":
    
    report = mne.Report()

    # decoding
    all_results = list()
    results = list()
    # print(subjects[0])
    # out = _decod_one_subject(subjects[0], add_is_sound_only_flag=ADD_IS_SOUND_ONLY_FLAG)


    # (
    #     fig_evo,
    #     fig_decod,
    #     results,
    #     fig_evo_ph,
    #     fig_decod_ph,
    #     results_ph,
    # ) = out     # tuple unpacking 1 個tuple拆成多個var

    # report.add_figure(fig_evo, subjects[0], tags="evo_word")
    # report.add_figure(fig_decod, subjects[0], tags="word")
    # report.add_figure(fig_evo_ph, subjects[0], tags="evo_phoneme")
    # report.add_figure(fig_decod_ph, subjects[0], tags="phoneme")

    # report.save("decoding_subject0.html", open_browser=False, overwrite=True)

    # all_results.append(results)
    # all_results.append(results_ph)
    # print("done")


    for i in range(UNTIL_SUBJECT):
        print(f"processing subject {i}...")

        out = _decod_one_subject(subjects[i], 
                                 until_session=UNTIL_SESSION,
                                 until_task=UNTIL_TASK,
                                 add_is_sound_only_flag=ADD_IS_SOUND_ONLY_FLAG)
        if out is None:
            continue

        (
            fig_evo,
            fig_decod,
            results,
            fig_evo_ph,
            fig_decod_ph,
            results_ph,
        ) = out     # tuple unpacking 1 個tuple拆成多個var

        report.add_figure(fig_evo, subjects[i], tags="evo_word")
        report.add_figure(fig_decod, subjects[i], tags="word")
        report.add_figure(fig_evo_ph, subjects[i], tags="evo_phoneme")
        report.add_figure(fig_decod_ph, subjects[i], tags="phoneme")

        report.save("decoding.html", open_browser=False, overwrite=True)

        all_results.append(results)
        all_results.append(results_ph)
        print("done")

    pd.concat(all_results, ignore_index=True).to_csv("decoding_results.csv")
