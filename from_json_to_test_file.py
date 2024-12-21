import json
import pandas as pd
import uuid
import string
import random

def generate_random_id(length=22):
    """Generate a random ID similar to the example format"""
    characters = string.ascii_letters + string.digits + '-_'
    return ''.join(random.choice(characters) for _ in range(length))

def create_restaurant_mapping(restaurants):
    """Create a mapping of restaurant names to unique itemIDs"""
    mapping = {}
    for idx, name in enumerate(restaurants):
        itemID = generate_random_id()
        mapping[name] = {
            'itemID': itemID,
            'item_num': idx
        }
    return mapping

def convert_reviews_to_csv(input_json_path, output_csv_path, mapping_file_path):
    """Convert JSON reviews to CSV format with required fields"""
    # Read the JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create lists to store the data
    csv_data = {
        'userID': [],
        'itemID': [],
        'review': [],
        'rating': [],
        'user_num': [],
        'item_num': []
    }
    
    # Create restaurant mapping
    restaurant_names = [restaurant['restaurant_name'] for restaurant in data]
    restaurant_mapping = create_restaurant_mapping(restaurant_names)
    
    # Save restaurant mapping to a separate file
    with open(mapping_file_path, 'w', encoding='utf-8') as f:
        json.dump({name: info['itemID'] for name, info in restaurant_mapping.items()}, 
                 f, ensure_ascii=False, indent=4)
    
    # Generate a set of unique userIDs first (one for each review)
    total_reviews = sum(len(restaurant['reviews']) for restaurant in data)
    unique_user_ids = set()
    while len(unique_user_ids) < total_reviews:
        unique_user_ids.add(generate_random_id())
    unique_user_ids = list(unique_user_ids)
    
    user_counter = 0
    
    # Process each restaurant and its reviews
    for restaurant in data:
        restaurant_name = restaurant['restaurant_name']
        itemID = restaurant_mapping[restaurant_name]['itemID']
        item_num = restaurant_mapping[restaurant_name]['item_num']
        
        # Process each review
        for review in restaurant['reviews']:
            # Check if rating is NaN or None
            if pd.isna(restaurant['rating']):
                continue
            
            userID = unique_user_ids[user_counter]
            
            csv_data['userID'].append(userID)
            csv_data['itemID'].append(itemID)
            csv_data['review'].append(review)
            csv_data['rating'].append(restaurant['rating'])
            csv_data['user_num'].append(user_counter)
            csv_data['item_num'].append(item_num)
            
            user_counter += 1
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)
    
    print(f"Created CSV file with {len(df)} reviews")
    print(f"Number of unique restaurants: {len(restaurant_mapping)}")
    print(f"Restaurant mapping saved to: {mapping_file_path}")


if __name__ == "__main__":
    input_json = "translated_combined_restaurant_data.json"
    output_csv = "restaurant_reviews.csv"
    mapping_file = "restaurant_name_ID_mapping.json"
    
    convert_reviews_to_csv(input_json, output_csv, mapping_file)