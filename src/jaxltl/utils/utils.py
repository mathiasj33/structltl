from joblib import Memory

from jaxltl import CACHE_DIR

memory = Memory(CACHE_DIR, verbose=0)
