import pandas as pd
import json
import os

def load_and_update_data(json_files, top_csv_path, low_csv_path):
    """
    1. Load new data from multiple JSON files.
    2. Determine new top vs. low performers.
    3. Append and save updates to existing CSV files.
    4. Return updated dataframes for top and low performers.
    """

    # Load new data
    new_posts = []
    for json_file in json_files:
        if not os.path.isfile(json_file):
            print(f"File not found: {json_file}")
            continue
        
        with open(json_file, 'r') as file:
            new_data = json.load(file)
            for post in new_data:
                new_posts.append({
                    'id': post['id'],
                    'message': post.get('message', ''),
                    'created_time': post['created_time'],
                    'engagement_rate': post.get('reactions', {}).get('summary', {}).get('total_count', 0)
                                       + post.get('comments', {}).get('summary', {}).get('total_count', 0)
                                       + post.get('shares', {}).get('count', 0),
                    'reactions': post.get('reactions', {}).get('summary', {}).get('total_count', 0),
                    'comments': post.get('comments', {}).get('summary', {}).get('total_count', 0),
                    'shares': post.get('shares', {}).get('count', 0),
                    'is_recipe': any(keyword in post.get('message', '').lower()
                                     for keyword in ['ingredients', 'directions', 'recipe',
                                                     'preheat', 'servings', 'bake', 'cook'])
                })

    new_data_df = pd.DataFrame(new_posts)

    # Filter to include only recipe posts
    new_data_df = new_data_df[new_data_df['is_recipe'] == True]

    # Load existing CSVs
    try:
        top_performers = pd.read_csv(top_csv_path)
    except FileNotFoundError:
        top_performers = pd.DataFrame()

    try:
        low_performers = pd.read_csv(low_csv_path)
    except FileNotFoundError:
        low_performers = pd.DataFrame()

    if new_data_df.empty:
        print("No new recipe posts found. Returning existing data.")
        return top_performers, low_performers

    # Split new data
    threshold = new_data_df['engagement_rate'].median()
    new_top_performers = new_data_df[new_data_df['engagement_rate'] > threshold]
    new_low_performers = new_data_df[new_data_df['engagement_rate'] <= threshold]

    # Append
    top_performers = pd.concat([top_performers, new_top_performers], ignore_index=True)
    low_performers = pd.concat([low_performers, new_low_performers], ignore_index=True)

    # Remove duplicates based on 'id'
    top_performers.drop_duplicates(subset='id', keep='last', inplace=True)
    low_performers.drop_duplicates(subset='id', keep='last', inplace=True)

    # Save CSVs
    top_performers.to_csv(top_csv_path, index=False)
    low_performers.to_csv(low_csv_path, index=False)

    print(f"Processed {len(json_files)} JSON file(s).")
    print(f"Updated top_performers.csv with {len(new_top_performers)} new posts.")
    print(f"Updated low_performers.csv with {len(new_low_performers)} new posts.")

    return top_performers, low_performers
