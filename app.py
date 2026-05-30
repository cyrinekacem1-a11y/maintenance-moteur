from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ============================================================
# تحميل الموديل
# ============================================================
model  = joblib.load('model_moteur.pkl')
scaler = joblib.load('scaler_moteur.pkl')

LABELS = {
    0: 'Normal',
    1: 'Surchauffe',
    2: 'Defaillance'
}

ACTIONS = {
    0: 'Moteur en bon état — continuer surveillance',
    1: 'Réduire charge — vérifier ventilation',
    2: 'ARRÊT IMMÉDIAT — maintenance requise'
}

COLORS = {
    0: 'green',
    1: 'orange',
    2: 'red'
}

# ============================================================
# Route principale — Prédiction
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, str):
            import json
            data = json.loads(data)
        # قراءة القيم
        courant     = float(data['courant'])
        vib_x       = float(data['vibration_x'])
        vib_y       = float(data['vibration_y'])
        temperature = float(data['temperature'])

        # Feature engineering
        vib_rms = np.sqrt((vib_x**2 + vib_y**2) / 2)
        vib_mag = np.sqrt(vib_x**2 + vib_y**2)

        # Préparation features
        features = np.array([[courant, vib_x, vib_y, vib_rms, vib_mag, temperature]])
        features_scaled = scaler.transform(features)

        # Prédiction
        pred  = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        return jsonify({
            'status'    : 'success',
            'etat'      : LABELS[pred],
            'label'     : int(pred),
            'confiance' : round(float(proba[pred]) * 100, 1),
            'action'    : ACTIONS[pred],
            'color'     : COLORS[pred]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ============================================================
# Route test — pour vérifier que l'API marche
# ============================================================
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'message': 'API Maintenance Prédictive opérationnelle ✅'})


# ============================================================
# Route test rapide avec valeurs dans l'URL
# ============================================================
@app.route('/test', methods=['GET'])
def test():
    exemples = [
        {'courant': 0.45, 'vibration_x': 0.01, 'vibration_y': 0.01, 'temperature': 38},
        {'courant': 0.55, 'vibration_x': 0.015,'vibration_y': 0.015,'temperature': 70},
        {'courant': 0.85, 'vibration_x': 0.09, 'vibration_y': 0.08, 'temperature': 56},
    ]
    resultats = []
    for ex in exemples:
        vib_rms = np.sqrt((ex['vibration_x']**2 + ex['vibration_y']**2) / 2)
        vib_mag = np.sqrt(ex['vibration_x']**2 + ex['vibration_y']**2)
        features = np.array([[ex['courant'], ex['vibration_x'], ex['vibration_y'], vib_rms, vib_mag, ex['temperature']]])
        features_scaled = scaler.transform(features)
        pred  = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        resultats.append({
            'input'     : ex,
            'etat'      : LABELS[pred],
            'confiance' : round(float(proba[pred]) * 100, 1),
            'action'    : ACTIONS[pred]
        })
    return jsonify(resultats)


if __name__ == '__main__':
    print("🚀 API démarrée sur http://localhost:5000")
    print("📡 Routes disponibles:")
    print("   GET  /ping   → test connexion")
    print("   GET  /test   → test 3 exemples")
    print("   POST /predict → prédiction réelle")
    app.run(debug=True, host='0.0.0.0', port=5000)
