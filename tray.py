import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import logging
import json
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train_model():
    try:
        if not os.path.exists('food_data_processed.csv'):
            raise FileNotFoundError("food_data_processed.csv not found. Run pre.py first.")
        df = pd.read_csv('food_data_processed.csv')
        logging.info(f"Loaded DataFrame with {len(df)} rows")

        # Ensure ingredients_list is a list
        df['ingredients_list'] = df['ingredients_list'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )
        # Ensure contextual_tags is a list
        df['contextual_tags'] = df['contextual_tags'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )

        df['combined_features'] = (
            df['ingredients'].fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['contextual_tags'].apply(lambda x: ' '.join(x)) + ' ' +
            df['origin'].fillna('') + ' ' +
            df['flavorProfile'].fillna('') + ' ' +
            df['spiceLevel'].fillna('')
        )

        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        model_components = {
            'tfidf': tfidf,
            'cosine_sim': cosine_sim,
            'df': df,
            'tfidf_matrix': tfidf_matrix
        }
        joblib.dump(model_components, 'food_recommender_model.joblib')
        logging.info("Model trained and saved to food_recommender_model.joblib")
        return model_components
    except Exception as e:
        logging.error(f"Error in training: {e}")
        return None

def get_recommendations(name, dietary_filters=None, category_filter=None, contextual_filter=None, origin=None, 
                       include_ingredients=None, exclude_ingredients=None, cosine_sim=None, df=None, top_n=5):
    try:
        filtered_df = df[df['available_in_menu'] == 1].copy()
        
        # Ensure ingredients_list and contextual_tags are lists
        filtered_df['ingredients_list'] = filtered_df['ingredients_list'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )
        filtered_df['contextual_tags'] = filtered_df['contextual_tags'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )
        
        if name.lower() not in filtered_df['name'].str.lower().values:
            logging.warning(f"Food '{name}' not found in dataset.")
            return None

        idx = filtered_df.index[filtered_df['name'].str.lower() == name.lower()][0]
        
        # Get the category of the input food for better recommendations
        input_category = filtered_df.loc[idx, 'categoryId']
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        indices = [i[0] for i in sim_scores[1:100]]  # Increased from 50 to 100 for better filtering

        filtered_df = filtered_df.iloc[indices].copy()
        
        # Category-based filtering with priority
        if category_filter:
            # First try exact category match
            category_filtered = filtered_df[filtered_df['categoryId'] == category_filter]
            if len(category_filtered) >= top_n:
                filtered_df = category_filtered
            else:
                # If not enough exact matches, include similar categories
                # Breakfast (1) can include snacks, Lunch/Dinner (2,3) are interchangeable
                if category_filter == 1:  # Breakfast
                    filtered_df = filtered_df[filtered_df['categoryId'].isin([1])]
                elif category_filter in [2, 3]:  # Lunch or Dinner
                    filtered_df = filtered_df[filtered_df['categoryId'].isin([2, 3])]
        else:
            # If no category filter specified, prioritize same category as input
            same_category = filtered_df[filtered_df['categoryId'] == input_category]
            other_category = filtered_df[filtered_df['categoryId'] != input_category]
            
            # Take 70% from same category, 30% from others
            same_count = min(int(top_n * 0.7), len(same_category))
            other_count = top_n - same_count
            
            filtered_df = pd.concat([
                same_category.head(same_count),
                other_category.head(other_count)
            ])

        # Apply other filters
        if dietary_filters:
            for restriction in dietary_filters:
                filtered_df = filtered_df[filtered_df[restriction] == 1]
        if origin:
            filtered_df = filtered_df[filtered_df['origin'] == origin]
        if include_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: any(ing.lower() in [i.lower() for i in x] for ing in include_ingredients)
            )]
        if exclude_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: not any(ing.lower() in [i.lower() for i in x] for ing in exclude_ingredients)
            )]
        if contextual_filter:
            filtered_df = filtered_df[filtered_df['contextual_tags'].apply(
                lambda x: contextual_filter in x
            )]

        # Sort by rating for better recommendations
        filtered_df = filtered_df.sort_values(['rating'], ascending=[False])
        
        recommendations = filtered_df.head(top_n)[['name', 'description', 'calories', 'protein', 'carbs', 'fats', 'price_LKR', 'categoryId', 'origin', 'rating']].to_dict('records')
        return recommendations if recommendations else None
    except Exception as e:
        logging.error(f"Error in get_recommendations: {e}")
        return None

def get_category_recommendations(category_filter, dietary_filters=None, origin=None, 
                               include_ingredients=None, exclude_ingredients=None, 
                               cosine_sim=None, df=None, tfidf=None, tfidf_matrix=None, top_n=10):
    """
    Get recommendations based purely on category and filters without requiring a specific food name
    """
    try:
        filtered_df = df[df['available_in_menu'] == 1].copy()
        
        # Ensure ingredients_list and contextual_tags are lists
        filtered_df['ingredients_list'] = filtered_df['ingredients_list'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )
        filtered_df['contextual_tags'] = filtered_df['contextual_tags'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )

        # Category mapping
        category_names = {1: 'Breakfast', 2: 'Lunch', 3: 'Dinner'}
        category_name = category_names.get(category_filter, 'Unknown')
        
        logging.info(f"Getting {category_name} recommendations with filters: dietary={dietary_filters}, origin={origin}")

        # Filter by category
        if category_filter:
            filtered_df = filtered_df[filtered_df['categoryId'] == category_filter]

        # Apply dietary filters
        if dietary_filters:
            for restriction in dietary_filters:
                filtered_df = filtered_df[filtered_df[restriction] == 1]
        
        # Apply origin filter
        if origin:
            filtered_df = filtered_df[filtered_df['origin'] == origin]
        
        # Apply ingredient filters
        if include_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: any(ing.lower() in [i.lower() for i in x] for ing in include_ingredients)
            )]
        if exclude_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: not any(ing.lower() in [i.lower() for i in x] for ing in exclude_ingredients)
            )]

        if filtered_df.empty:
            logging.warning(f"No {category_name} items found after filters.")
            return None

        # Create query for semantic similarity
        query_parts = [category_name]
        if origin:
            query_parts.append(origin)
        if dietary_filters:
            query_parts.extend(dietary_filters)
        if include_ingredients:
            query_parts.extend(include_ingredients)
        
        query = ' '.join(query_parts)
        query_vector = tfidf.transform([query])
        sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Add similarity scores to dataframe
        filtered_df['similarity'] = sim_scores[filtered_df.index]
        
        # Sort by similarity and rating
        filtered_df = filtered_df.sort_values(['similarity', 'rating'], ascending=[False, False])
        
        recommendations = filtered_df.head(top_n)[['name', 'description', 'calories', 'protein', 'carbs', 'fats', 'price_LKR', 'categoryId', 'origin', 'rating']].to_dict('records')
        return recommendations if recommendations else None
        
    except Exception as e:
        logging.error(f"Error in get_category_recommendations: {e}")
        return None

def get_meal_plan_with_budget(calorie_goal, carb_goal, budget, dietary_filters=None, origin=None, meal_type=None, 
                              include_ingredients=None, exclude_ingredients=None, cosine_sim=None, df=None, tfidf=None, 
                              tfidf_matrix=None, top_n=5):
    try:
        filtered_df = df[df['available_in_menu'] == 1].copy()
        logging.info(f"Generating meal plan: calories={calorie_goal}, carbs={carb_goal}, budget={budget}, dietary={dietary_filters}, origin={origin}, meal_type={meal_type}")

        # Ensure ingredients_list and contextual_tags are lists
        filtered_df['ingredients_list'] = filtered_df['ingredients_list'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )
        filtered_df['contextual_tags'] = filtered_df['contextual_tags'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x if isinstance(x, list) else []
        )

        meal_types = {
            'Breakfast': {'categoryIds': [1], 'calorie_share': 0.3, 'carb_share': 0.3},
            'Lunch': {'categoryIds': [2], 'calorie_share': 0.4, 'carb_share': 0.4},
            'Dinner': {'categoryIds': [3], 'calorie_share': 0.3, 'carb_share': 0.3},
            'Snack': {'categoryIds': [1], 'calorie_share': 0.2, 'carb_share': 0.2}
        }

        if meal_type not in meal_types:
            meal_type = 'Lunch'

        filtered_df = filtered_df[filtered_df['categoryId'].isin(meal_types[meal_type]['categoryIds'])]
        if dietary_filters:
            for restriction in dietary_filters:
                filtered_df = filtered_df[filtered_df[restriction] == 1]
        if origin:
            filtered_df = filtered_df[filtered_df['origin'] == origin]
        if include_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: any(ing.lower() in [i.lower() for i in x] for ing in include_ingredients)
            )]
        if exclude_ingredients:
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(
                lambda x: not any(ing.lower() in [i.lower() for i in x] for ing in exclude_ingredients)
            )]

        cal_limit = calorie_goal * meal_types[meal_type]['calorie_share'] * 1.5
        carb_limit = carb_goal * meal_types[meal_type]['carb_share'] * 1.5
        filtered_df = filtered_df[
            (filtered_df['calories'] <= cal_limit) &
            (filtered_df['carbs'] <= carb_limit) &
            (filtered_df['price_LKR'] <= budget)
        ]

        if filtered_df.empty:
            logging.warning("No meals found after filters.")
            return {'message': f"No {meal_type} meals found within your budget and preferences! Try adjusting your filters. 😊", 'meals': [], 'subtotal': 0}

        query = f"{meal_type} {origin or ''} {' '.join(dietary_filters or [])} {' '.join(include_ingredients or [])}"
        query_vector = tfidf.transform([query])
        sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        filtered_df['similarity'] = sim_scores[filtered_df.index]
        filtered_df = filtered_df.sort_values(['similarity', 'rating'], ascending=[False, False])

        meals = filtered_df.head(top_n)[['name', 'description', 'calories', 'carbs', 'protein', 'fats', 'price_LKR', 'categoryId', 'origin', 'rating']].to_dict('records')
        subtotal = sum(meal['price_LKR'] for meal in meals)
        return {
            'message': f"Here's your {meal_type} meal plan within LKR {budget}! Add or remove meals to customize. 🍽️",
            'meals': meals,
            'subtotal': subtotal
        }
    except Exception as e:
        logging.error(f"Error in get_meal_plan_with_budget: {e}")
        return {'message': f"Oops, something broke while planning your meal! Please try again. 😅", 'meals': [], 'subtotal': 0}

if __name__ == '__main__':
    model_components = train_model()
    if not model_components:
        logging.error("Model training failed. Exiting.")
        exit(1)