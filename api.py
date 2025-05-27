from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import logging
import numpy as np
from tray import get_meal_plan_with_budget, get_recommendations, get_category_recommendations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={
    '/meal_plan': {'origins': '*'}, 
    '/recommendations': {'origins': '*'},
    '/category_recommendations': {'origins': '*'}
})

try:
    model_components = joblib.load('food_recommender_model.joblib')
    df = model_components['df']
    cosine_sim = model_components['cosine_sim']
    tfidf = model_components['tfidf']
    tfidf_matrix = model_components['tfidf_matrix']
except FileNotFoundError:
    logging.error("Error: food_recommender_model.joblib not found. Run tray.py first.")
    exit(1)

def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

@app.route('/recommendations', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True)
        name = data.get('name')
        if not name:
            return jsonify({'error': 'Food name required'}), 400

        recommendations = get_recommendations(
            name=name,
            dietary_filters=data.get('dietary_filters', []),
            category_filter=data.get('category_filter'),
            origin=data.get('origin'),
            include_ingredients=data.get('include_ingredients', []),
            exclude_ingredients=data.get('exclude_ingredients', []),
            cosine_sim=cosine_sim,
            df=df,
            top_n=10
        )
        if not recommendations:
            return jsonify({'message': f'No suggestions found for {name}. Try another food! 😊', 'foods': []})
        return jsonify({
            'message': f'Here are some tasty suggestions for {name}! 😋',
            'foods': convert_to_serializable(recommendations)
        })
    except Exception as e:
        logging.error(f"Error in recommendations: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/category_recommendations', methods=['POST'])
def category_recommend():
    try:
        data = request.get_json(force=True)
        category_filter = data.get('category_filter')
        if not category_filter:
            return jsonify({'error': 'Category filter required (1=Breakfast, 2=Lunch, 3=Dinner)'}), 400

        category_names = {1: 'Breakfast', 2: 'Lunch', 3: 'Dinner'}
        category_name = category_names.get(category_filter, 'Unknown')

        recommendations = get_category_recommendations(
            category_filter=category_filter,
            dietary_filters=data.get('dietary_filters', []),
            origin=data.get('origin'),
            include_ingredients=data.get('include_ingredients', []),
            exclude_ingredients=data.get('exclude_ingredients', []),
            cosine_sim=cosine_sim,
            df=df,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            top_n=data.get('top_n', 10)
        )
        
        if not recommendations:
            return jsonify({
                'message': f'No {category_name} recommendations found with your preferences. Try adjusting your filters! 😊', 
                'foods': []
            })
        
        return jsonify({
            'message': f'Here are some great {category_name} recommendations for you! 🍽️',
            'foods': convert_to_serializable(recommendations)
        })
    except Exception as e:
        logging.error(f"Error in category_recommendations: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/meal_plan', methods=['POST'])
def meal_plan():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'Invalid input data'}), 400

        meal_plan = get_meal_plan_with_budget(
            calorie_goal=data.get('calorie_goal', 1000),
            carb_goal=data.get('carb_goal', 80),
            budget=data.get('budget', 1000),
            dietary_filters=data.get('dietary_filters', []),
            include_ingredients=data.get('include_ingredients', []),
            exclude_ingredients=data.get('exclude_ingredients', []),
            origin=data.get('origin'),
            meal_type=data.get('meal_type', 'Lunch'),
            cosine_sim=cosine_sim,
            df=df,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            top_n=10
        )
        return jsonify(convert_to_serializable(meal_plan))
    except Exception as e:
        logging.error(f"Error in meal_plan: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Smart Meal Planner API is running!',
        'endpoints': {
            '/recommendations': 'Get recommendations based on a specific food name',
            '/category_recommendations': 'Get recommendations based on meal category',
            '/meal_plan': 'Get a complete meal plan with budget constraints'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)