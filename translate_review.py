import json
import re
from googletrans import Translator
import time

def remove_emoticons(text):
    """
    Remove emoticons, emojis and other special characters from text
    while preserving Korean and English text
    """
    # Remove emoticons like (*≧∇≦)ﾉ but preserve text inside other parentheses
    # text = re.sub(r'\([^가-힣a-zA-Z0-9]*\)', '', text)
    
    # Pattern to match emojis and specific symbols
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  # emoticons
        # u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        # u"\U0001F680-\U0001F6FF"  # transport & map symbols
        # u"\U00002702-\U000027B0"  # dingbats
        # u"\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE)
    
    # Remove emojis and special characters
    text = emoji_pattern.sub('', text)
    
    # Additional special character removal while ensuring Korean characters are intact
    text = re.sub(r'[~♡!^]+', '', text)  # Keep or adjust characters like `ㅎ` and `ㅋ` if stylistic Korean
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leftover empty parentheses
    text = re.sub(r'\(\s*\)', '', text)
    
    return text.strip()

def translate_reviews(input_file, output_file):
    """
    Read from JSON file, translate Korean reviews to English and save to a new JSON file
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Initialize translator
        translator = Translator()
        
        # Print first restaurant's first review for demonstration
        if input_data and len(input_data) > 0 and 'reviews' in input_data[0] and len(input_data[0]['reviews']) > 0:
            first_review = input_data[0]['reviews'][0]
            print("\nOriginal first review:")
            print(first_review)
            
            cleaned_text = remove_emoticons(first_review)
            print("\nAfter cleaning:")
            print(cleaned_text)
            
            try:
                translation = translator.translate(cleaned_text, src='ko', dest='en')
                if translation and translation.text:
                    print("\nAfter translation:")
                    print(translation.text)
            except Exception as e:
                print(f"\nTranslation error: {str(e)}")
        
        print("\nContinuing with full translation...")
        
        # Rest of the code remains the same...
        output_data = []
        
        # Process each restaurant
        for idx, restaurant in enumerate(input_data):
            print(f"\nProcessing restaurant {idx + 1}/{len(input_data)}: {restaurant['restaurant_name']}")
            restaurant_data = restaurant.copy()
            translated_reviews = []
            
            for review_idx, review in enumerate(restaurant['reviews']):
                cleaned_text = remove_emoticons(review)
                
                if not cleaned_text.strip():
                    translated_reviews.append("")
                    continue
                
                try:
                    time.sleep(2)
                    translation = translator.translate(cleaned_text, src='ko', dest='en')
                    if translation and translation.text:
                        translated_reviews.append(translation.text)
                        print(f"Translated review {review_idx + 1}/{len(restaurant['reviews'])}")
                    else:
                        translated_reviews.append(cleaned_text)
                        print(f"Warning: Empty translation for review {review_idx + 1}")
                    
                except Exception as e:
                    print(f"Error translating review {review_idx + 1}: {str(e)}")
                    translated_reviews.append(cleaned_text)
            
            restaurant_data['reviews'] = translated_reviews
            output_data.append(restaurant_data)
            
            if (idx + 1) % 5 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"Progress saved after restaurant {idx + 1}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        print(f"\nTranslation completed. Output saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if output_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            print("Partial progress has been saved.")

if __name__ == "__main__":
    input_file = "updated_combined_restaurant_data.json"  
    output_file = "translated_combined_restaurant_data.json"
    
    translate_reviews(input_file, output_file)