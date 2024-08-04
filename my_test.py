from wordfreq import zipf_frequency

word_list=[
    "a", "an", "the", "is", "are", "I", "it", "he", "she", "they", "we",
    "english", "american", "good", "bad", "new", "great", "best",
    "electronic", "device", "sesquipedalian"
]

for wd in word_list:
    print(f"frequency of {wd} is {zipf_frequency(wd, "en")}")