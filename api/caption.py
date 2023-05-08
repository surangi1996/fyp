import pandas as pd
import random

df = pd.read_csv('data/caption.csv')

captions_dict = df.groupby('sport')['caption'].apply(list).to_dict()

def get_caption(sport):
    caption_arr = captions_dict[sport]
    return random.choice(caption_arr)