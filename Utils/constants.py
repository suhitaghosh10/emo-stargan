COS = 'cos'
PEARSON = 'pearson'
L1_LOSS = 'l1'
DOT = 'dot'
DTW = 'soft-dtw'

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

EMO_DICT = {
    "Surprise" : 0,
    "Sad" : 1,
    "Neutral" : 2,
    "Happy" : 3,
    "Angry" : 4
}

EMO_DB_DICT = {
    "W" : 0,
    "F" : 1,
    "T" : 2,
    "N" : 3
}