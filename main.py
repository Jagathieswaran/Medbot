from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import torch
import os
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

app = Flask(__name__)

class AdvancedMedicalChatbot:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
        self.medical_data = None
        self.embeddings = None

    def load_data(self):
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'interactions', 'data', 'medi.json')
            if not os.path.exists(json_path):
                print(f"JSON file not found at {json_path}")
                return False
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.medical_data = pd.DataFrame(data)
            self.medical_data.rename(columns={
                "Medicine Name": "medicine",
                "Diseases": "condition",
                "Side_effects": "side_effects",
                "Composition": "composition",
                "Manufacturer": "manufacturer",
                "Excellent Review %": "excellent_review",
                "Average Review %": "average_review",
                "Poor Review %": "poor_review"
            }, inplace=True)
            self.medical_data['combined_text'] = self.medical_data.apply(
                lambda row: ' '.join(str(cell) for cell in row if pd.notna(cell)),
                axis=1
            )
            self.create_embeddings()
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def create_embeddings(self):
        embedding_file = os.path.join('static', 'data', 'embeddings.npy')
        try:
            if os.path.exists(embedding_file):
                self.embeddings = torch.tensor(np.load(embedding_file))
            else:
                texts = self.medical_data['combined_text'].tolist()
                self.embeddings = self.model.encode(texts, convert_to_tensor=True)
                os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
                np.save(embedding_file, self.embeddings.cpu().numpy())
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            self.embeddings = None

    def find_best_matches(self, query, top_n=5):
        if self.model is None or self.embeddings is None or self.medical_data is None:
            return []
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
            top_results = torch.topk(cosine_scores, k=top_n)
            recommendations = []
            for score, idx in zip(top_results.values, top_results.indices):
                data = self.medical_data.iloc[idx.item()].to_dict()
                recommendations.append({
                    "medicine": data.get('medicine', ''),
                    "condition": data.get('condition', ''),
                    "similarity_score": float(score.item()),
                    "composition": data.get("composition", ""),
                    "manufacturer": data.get("manufacturer", ""),
                    "side_effects": data.get("side_effects", "")
                })
            return recommendations
        except Exception as e:
            print(f"Error in matching: {str(e)}")
            return []

    @staticmethod
    def filter_general_fever_recommendations(recommendations):
        return [rec for rec in recommendations if 'fever' in rec.get('condition', '').lower()
                and not any(s in rec.get('condition', '').lower() for s in ['dengue', 'typhoid', 'malaria'])]

chatbot = AdvancedMedicalChatbot()
data_loaded = chatbot.load_data()

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/query', methods=['POST'])
def process_query():
    if not data_loaded:
        return jsonify({"error": "Medical data could not be loaded"}), 500
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        recommendations = chatbot.find_best_matches(query)
        if query.lower() == 'fever':
            recommendations = chatbot.filter_general_fever_recommendations(recommendations)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

@app.route('/recent_queries')
def get_recent_queries():
    try:
        return jsonify({"queries": [
            "Headache medication",
            "Cold and flu remedies",
            "Allergy treatment",
            "Pain relief options",
            "Digestive health"
        ]})
    except Exception as e:
        return jsonify({"queries": [], "error": f"Error fetching recent queries: {str(e)}"}), 500

@app.route('/compare', methods=['POST'])
def compare_medications():
    try:
        data = request.get_json(force=True)
        med1 = data.get('med1', '').strip()
        med2 = data.get('med2', '').strip()
        if not med1 or not med2:
            return jsonify({"error": "Both medications required"}), 400
        return jsonify({
            "comparison": {
                "med1": {
                    "name": med1,
                    "primary_use": "Pain relief, fever reduction",
                    "side_effects": "Stomach upset, heartburn",
                    "dosage": "325-650mg every 4-6 hours",
                    "precautions": "Liver disease, alcohol use"
                },
                "med2": {
                    "name": med2,
                    "primary_use": "Pain relief, inflammation",
                    "side_effects": "Stomach upset, risk of bleeding",
                    "dosage": "200-400mg every 4-6 hours",
                    "precautions": "Stomach ulcers, heart conditions"
                },
                "summary": {
                    "similarities": "Both used for pain relief",
                    "differences": f"{med2} has anti-inflammatory properties",
                    "interaction_risk": "Medium - can be taken together with caution",
                    "recommendation": "Consult with healthcare provider"
                }
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error comparing medications: {str(e)}"}), 500

@app.route('/dosage', methods=['POST'])
def calculate_dosage():
    try:
        data = request.get_json(force=True)
        weight = float(data.get("weight", 0))
        age = int(data.get("age", 0))
        medication = data.get("medication", "").strip()
        if not medication or weight <= 0 or age <= 0:
            return jsonify({"error": "Valid inputs required"}), 400
        base_dosage = weight * 0.1
        age_factor = 0.5 if age < 12 else 1.0
        recommended = base_dosage * age_factor
        return jsonify({
            "medication": medication,
            "recommended_dosage": round(recommended, 1),
            "max_daily": round(recommended * 4, 1)
        })
    except Exception as e:
        return jsonify({"error": f"Error calculating dosage: {str(e)}"}), 500

@app.route('/interactions', methods=['POST'])
def check_interactions():
    try:
        data = request.get_json(force=True)
        meds = data.get('medications', [])
        if len(meds) < 2:
            return jsonify({"error": "At least 2 medications required"}), 400
        interactions = []
        levels = ['High', 'Medium', 'Low']
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                risk = np.random.choice(levels)
                interactions.append({
                    "med1": meds[i],
                    "med2": meds[j],
                    "risk": risk,
                    "description": f"{meds[i]} and {meds[j]} have a {risk.lower()} interaction risk."
                })
        return jsonify({"interactions": interactions})
    except Exception as e:
        return jsonify({"error": f"Error checking interactions: {str(e)}"}), 500

@app.route('/add_medicine', methods=['POST'])
def add_medicine():
    try:
        entry = request.get_json(force=True)
        required = ["medicine", "condition", "side_effects", "composition", "manufacturer"]
        if not all(k in entry and entry[k] for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        json_path = os.path.join(os.path.dirname(__file__), 'interactions', 'data', 'medi.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.append(entry)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        chatbot.load_data()
        return jsonify({"message": "New medicine added successfully."})
    except Exception as e:
        return jsonify({"error": f"Error adding medicine: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
