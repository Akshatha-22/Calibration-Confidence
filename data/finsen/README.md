# Create README in raw folder
echo "# FinSen Raw Data Files

## File 1: [filename1.csv]
- Description: [what this file contains]
- Columns: [list columns]
- Rows: [number of rows]

## File 2: [filename2.csv]
- Description: Stock price data
- Columns: date, open, high, low, close, volume
- Rows: 4,000+

## File 3: [filename3.csv]
- Description: News sentiment scores
- Columns: date, sentiment_score, article_count
- Rows: 4,000+

## Date Range: 2007-01-01 to 2023-12-31
" > data/finsen/raw/README.md