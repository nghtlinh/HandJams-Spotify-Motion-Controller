import os
import numpy as np

TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing user-library-modify user-library-read"

THRESHOLD = 0.995
SEQUENCE_LENGTH = 30
NO_SEQUENCES = 66
DATA_PATH = os.path.join('MP_DATA_FUNC')
ACTIONS = np.array(['next', 'prev', 'love', 'louder', 'quieter', 'play_pause'])