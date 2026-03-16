import pandas as pd
from collections import Counter
import itertools
chunks = pd.read_csv('data/finsen/processed/train.csv', usecols=['Tag','Category','content_length'], chunksize=100000)
cat_counts = Counter()
tag_counts = Counter()
lengths = []
for chunk in itertools.islice(chunks, 10):
    cat_counts.update(chunk['Category'].dropna())
    tag_counts.update(chunk['Tag'].dropna())
    lengths.extend(chunk['content_length'].dropna().astype(int).tolist())
print('Sampled rows', len(lengths))
print('Top categories', cat_counts.most_common(5))
print('Top tags', tag_counts.most_common(5))
import statistics
print('Content length mean', statistics.mean(lengths))
print('Min', min(lengths), 'Max', max(lengths))
print('Percentiles', {p: statistics.quantiles(lengths, n=100)[p-1] for p in (10,25,50,75,90)})
