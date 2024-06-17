import pandas as pd
import glob
import os

# load kaggle data
th_text = []
en_text = []

with open("data/kaggle/all.tok/all.tok.en", "r") as f:
    en_text = [line.strip() for line in f]
    
with open("data/kaggle/all.tok/all.tok.th", "r") as f:
    th_text = [line.strip() for line in f]

en_ser = pd.Series(en_text).map(lambda x: x.replace(' ','').lower()).drop_duplicates()

# Check Matching Data
stats = []
total_csv_row = 0
total_match_row = 0

total_match = []

for path in glob.glob('./data/scb-mt-en-th-2020+mt-opus/*.csv'):
    _df = pd.read_csv(path)
    en = _df['en_text'].astype(str).drop_duplicates()
    
    matched = en_ser[en_ser.isin(en.map(lambda x: x.replace(' ', '').lower()))]
    unmatched = en_ser[~en_ser.isin(en.map(lambda x: x.replace(' ', '').lower()))]
    
    total_match += matched.to_list()
    
    stats.append({
        'fname': os.path.basename(path),
        'len_csv': len(_df),
        'len_matched': len(matched)
    })
    
    total_csv_row += len(_df)
    total_match_row += len(matched)
    

stats.append({
    'fname': 'unmatched',
    'len_csv': 0,
    'len_matched': len(en_ser) - total_match_row
})

stats.append({
    'fname': 'total',
    'len_csv': total_csv_row,
    'len_matched': len(en_ser)
})

stats = pd.DataFrame(stats)
print("== Stats ==")
print(stats)

en_ser_unmatch = en_ser[~en_ser.isin(total_match)].drop_duplicates()

new_en_text = pd.Series(en_text).iloc[en_ser_unmatch.index]
new_th_text = (pd.Series(th_text).iloc[en_ser_unmatch.index]).map(lambda x: x.replace(" ", ''))

df_unmatched = pd.DataFrame({
    'en_text': new_en_text,
    'th_text': new_th_text,
})

print("== New Unmatch Dataset ==")
print(df_unmatched)

df_unmatched.to_csv('./data/scb-mt-en-th-2020+mt-opus/unmatch_kaggle_corpus.csv')

