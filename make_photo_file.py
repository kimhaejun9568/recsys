import json
import random
import os
import requests
from urllib.parse import urlparse
import time

def create_photo_directory(base_dir="restaurant_photos"):
    """Create directory for storing photos if it doesn't exist"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def download_image(url, file_path):
    """Download image from URL and save to specified path"""
    try:
        response = requests.get(url, timeout=10)  # Added timeout
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False

def is_valid_rating(rating):
    """Check if rating is valid (not NaN)"""
    if rating is None:
        return False
    if isinstance(rating, float) and math.isnan(rating):
        return False
    return True

def process_all_restaurants(input_file, output_json, photo_dir, mapping_file, resume_from=None):
    """Process photos for all restaurants"""
    # Read the mapping file
    with open(mapping_file, 'r', encoding='utf-8') as f:
        restaurant_mapping = json.load(f)
    
    # Read the restaurant data
    with open(input_file, 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    
    
    if not restaurants:
        print("No restaurants found in the input file")
        return
    
    # List to store all photo entries
    photo_entries = []
    photo_id_counter = 1
    
    # Load existing entries if resuming
    if resume_from and os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            photo_entries = [json.loads(line) for line in f]
        photo_id_counter = max([int(entry['photo_id'].split('_')[1]) for entry in photo_entries]) + 1
        
    # Find starting point if resuming
    start_idx = 0
    if resume_from:
        for idx, restaurant in enumerate(restaurants):
            if restaurant['restaurant_name'] == resume_from:
                start_idx = idx
                break
    
    # Possible labels for photos
    labels = ["inside", "outside", "food", "drink"]
    processed_count = 0

    # Process each restaurant
    for idx, restaurant in enumerate(restaurants[start_idx:], start=start_idx):
        restaurant_name = restaurant['restaurant_name']

        # Skip if rating is NaN
        if not is_valid_rating(restaurant.get('rating')):
            print(f"\nSkipping restaurant {restaurant_name} due to invalid rating")
            continue

        # Get the business_id from mapping
        business_id = restaurant_mapping.get(restaurant_name)
        if not business_id:
            print(f"No mapping found for restaurant {restaurant_name}, skipping...")
            continue
        
        processed_count += 1
        print(f"\nProcessing restaurant {idx + 1}/{len(restaurants)}: {restaurant_name}")
        print(f"Using business_id: {business_id}")
        
        # Get all image URLs for this restaurant
        image_urls = restaurant.get('images', [])
        print(f"Total available images: {len(image_urls)}")
        
        # Randomly select 10 images (or all if less than 10)
        num_photos = min(10, len(image_urls))
        selected_urls = random.sample(image_urls, num_photos)
        
        # Process each selected image
        restaurant_photos = 0
         for url in selected_urls:
            photo_id = f"img_{photo_id_counter}"
            file_name = f"{photo_id}.jpg"
            file_path = os.path.join(photo_dir, file_name)
            
            print(f"Downloading image {photo_id} from URL: {url}")
            
            # Download the image
            if download_image(url, file_path):
                # Create entry for photo.json
                entry = {
                    "business_id": business_id,
                    "photo_id": photo_id,
                    "label": random.choice(labels)
                }
                photo_entries.append(entry)
                # Write the new entry immediately
                with open(output_json, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                print(f"Successfully downloaded and processed photo {photo_id}")
                restaurant_photos += 1
            else:
                print(f"Failed to download photo {photo_id}")
            
            photo_id_counter += 1
            # Add random delay between downloads
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"Processed {restaurant_photos} photos for {restaurant_name}")
        
        # Save progress periodically
        if processed_count % 5 == 0:
            print(f"Progress: {processed_count} valid restaurants processed")
    
    print(f"\nProcessing completed:")
    print(f"- Total restaurants processed: {processed_count}")
    print(f"- Total photos processed: {len(photo_entries)}")
    print(f"- Photos saved in directory: {photo_dir}")
    print(f"- Photo entries saved to {output_json}")


if __name__ == "__main__":
    # File paths
    input_file = "translated_combined_restaurant_data.json"
    mapping_file = "restaurant_name_ID_mapping.json"
    output_json = "photos.json"
    photo_dir = create_photo_directory()
    
    # If you need to resume from a specific restaurant, set its name here
    resume_from = None  # Or set to restaurant name to resume from
    
    # Process all restaurants
    process_all_restaurants(input_file, output_json, photo_dir, mapping_file, resume_from)