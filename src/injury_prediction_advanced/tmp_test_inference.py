from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import pipeline, parse_request_data

sample = {
    'Injury': 'Hamstring Strain',
    'player_age': '25',
    'player_position': 'Centre-Back',
    'club': 'FC Barcelona',
    'league': 'La Liga',
    'injury_from_parsed': '2024-02-15',
    'injury_until_parsed': '2024-03-01',
    'player_name': 'Test Player',
    'Season': '24/25',
}

df = parse_request_data(sample)
print('Parsed input:', df.to_dict(orient='records'))
preds, meta = pipeline.predict(df, return_metadata=True)
print('Prediction:', preds)
print('Meta:', meta)
