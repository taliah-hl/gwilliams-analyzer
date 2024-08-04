# install requirements in conda

if encoounter following error while installing by conda
```
PackagesNotFoundError: The following packages are not available from current channels:

  - wordfreq
  - mne_bids
  - nibabel
```

try the below

`conda install conda-forge::wordfreq`
`conda install --yes conda-forge/label/cf202003::mne-bids`

用conda 裝完mne_bids 還是有問題!