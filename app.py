from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime

# Load the model and label encoder
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

file_path = 'MOCK_DATA (4) - MOCK_DATA (4).csv (1) (1).csv'  # Path to the CSV file

app = Flask(__name__)

# Convert timestamp to day and slot number
def convert_timestamp_to_day_slot(timestamp):
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    day = dt.strftime('%A')
    
    hour = dt.hour
    
    if 10 <= hour < 22:
        slot_no = (hour - 10) + 1
    else:
        slot_no = 12  # Default to the last slot if outside defined slots

    return day, slot_no

@app.route('/predict_traffic', methods=['POST'])
def predict_traffic():
    data = request.get_json()
    id = data.get('id')
    timestamp = data.get('timestamp')
    
    if not id or not timestamp:
        return jsonify({'error': 'Missing id or timestamp'}), 400
    
    # Convert timestamp to day and slot number
    day, slot_no = convert_timestamp_to_day_slot(timestamp)
    day_encoded = label_encoder.transform([day])[0]
    
    # Load the dataset from CSV
    df = pd.read_csv("MOCK_DATA (4) - MOCK_DATA (4).csv (1) (1).csv")
    
    # Check if a record exists for the current id, day, and slot_no
    record = df[(df['id'] == id) & (df['day'] == day) & (df['slot_no.'] == slot_no)]
    
    if not record.empty:
        # If record exists, increment the Booked_slots by 1
        df.loc[record.index, 'Booked_slots'] += 1
        booked_slots = df.loc[record.index, 'Booked_slots'].values[0]
        total_slots = df.loc[record.index, 'Total_slots'].values[0]
    else:
        # If no record exists, add a new row with Booked_slots set to 1
        booked_slots = 1
        total_slots = 10  # Assuming a default total_slots value
        new_row = pd.DataFrame({
            'id': [id],
            'day': [day],
            'slot_no.': [slot_no],
            'Booked_slots': [booked_slots],
            'Total_slots': [total_slots],
            'day_encoded': [day_encoded]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated dataset back to the CSV file
    df.to_csv('MOCK_DATA (4) - MOCK_DATA (4).csv (1) (1).csv', index=False)
    
    # Prepare the input for the model
    input_data = pd.DataFrame({
        'day_encoded': [day_encoded],
        'slot_no.': [slot_no],
        'Booked_slots': [booked_slots],
        'Total_slots': [total_slots]
    })
    
    # Predict the traffic category
    traffic_prediction = model.predict(input_data)[0]
    
    return jsonify({
        'id': id,
        'day': day,
        'slot_no.': slot_no,
        'traffic_message': traffic_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
