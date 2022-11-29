"""
Utils
"""


import numpy as np

# NOTE: raw txt file has emotion labels 1-indexed, I am shifting to zero-indexing
emotion_lookup = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}
race_lookup = {
    0: "Caucasian",
    1: "African-American",
    2: "Asian",
}

def get_emotion(index: int):
    return emotion_lookup[index]

def get_race(index: int):
    return race_lookup[index]

def one_hot_encode(label, n_labels):
    """
    One-hot encode a label (0-indexed).
    """
    out = np.zeros((n_labels), dtype=np.float32)
    out[label] = 1
    return out
