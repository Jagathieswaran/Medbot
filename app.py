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
            print(f"Error loading SentenceTransformer model: {str(e)}")
            self.model = None
        self.medical_data = None
        self.embeddings = None

    def load_data(self):
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'interactions', 'data', 'medi.json')
            if not os.path.exists(json_path):
                print(f"JSON data file not found at {json_path}")
                return False
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.medical_data = pd.DataFrame(data)
            self.medical_data.rename(columns={
                "Medicine Name": "medicine",
                "Diseases": "condition",
                "Side_effects": "side_effects",
                "Composition": "composition",
                "Manufacturer": "manufacturer",
                "Excellent Review %": "excellent_review",
                "Average Review %": "average_review",
                "Poor Review %": "poor_review",
            }, inplace=True)
            self.medical_data['combined_text'] = self.medical_data.apply(
                lambda row: ' '.join(str(cell) for cell in row if pd.notna(cell)),
                axis=1
            )
            self.create_embeddings()
            return True
        except Exception as e:
            print(f"Error loading JSON data: {str(e)}")
            return False

    def create_embeddings(self):
        embedding_file = os.path.join('static', 'data', 'embeddings.npy')
        try:
            if os.path.exists(embedding_file):
                self.embeddings = torch.tensor(np.load(embedding_file))
            else:
                combined_texts = self.medical_data['combined_text'].tolist()
                self.embeddings = self.model.encode(combined_texts, convert_to_tensor=True)
                os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
                np.save(embedding_file, self.embeddings.cpu().numpy())
        except Exception as e:
            print(f"Error creating/loading embeddings: {str(e)}")
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
            print(f"Error in find_best_matches: {str(e)}")
            return []

    @staticmethod
    def filter_general_fever_recommendations(recommendations):
        general_fever_recommendations = []
        for rec in recommendations:
            condition = rec.get('condition', '').lower()
            if 'fever' in condition and not any(specific in condition for specific in ['dengue', 'typhoid', 'malaria']):
                general_fever_recommendations.append(rec)
        return general_fever_recommendations

chatbot = AdvancedMedicalChatbot()
data_loaded = chatbot.load_data()

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

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
        recent_queries = [
            "Headache medication",
            "Cold and flu remedies",
            "Allergy treatment",
            "Pain relief options",
            "Digestive health"
        ]
        return jsonify({"queries": recent_queries})
    except Exception as e:
        return jsonify({"queries": [], "error": f"Error fetching recent queries: {str(e)}"}), 500

@app.route('/api/v1/search', methods=['GET'])
def api_search():
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({"results": [], "error": "No query provided"}), 400
        recommendations = chatbot.find_best_matches(query)
        if query.lower() == 'fever':
            recommendations = chatbot.filter_general_fever_recommendations(recommendations)
        return jsonify({"results": recommendations})
    except Exception as e:
        return jsonify({"results": [], "error": f"Error in search: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)