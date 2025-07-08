from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load datasets with error handling
try:
    sym_des = pd.read_csv("datasets/symtom_df.csv")
    precautions = pd.read_csv("datasets/precautions_df.csv")
    workout = pd.read_csv("datasets/workout_df.csv")
    description = pd.read_csv("datasets/description.csv")
    medications = pd.read_csv('datasets/medications.csv')
    diets = pd.read_csv("datasets/diets.csv")

    # Load model
    svc = pickle.load(open('models/svc.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure all dataset files and model files are in the correct directories")

# Example symptoms dictionary - you should replace this with your actual data
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79,
    'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83,
    'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
    'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
    'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Example diseases list - you should replace this with your actual data
diseases_list = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    4: 'Drug Reaction', 5: 'Peptic ulcer diseae', 6: 'AIDS', 7: 'Diabetes',
    8: 'Gastroenteritis', 9: 'Bronchial Asthma', 10: 'Hypertension', 11: 'Migraine',
    12: 'Cervical spondylosis', 13: 'Paralysis (brain hemorrhage)', 14: 'Jaundice',
    15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid', 19: 'hepatitis A',
    20: 'Hepatitis B', 21: 'Hepatitis C', 22: 'Hepatitis D', 23: 'Hepatitis E',
    24: 'Alcoholic hepatitis', 25: 'Tuberculosis', 26: 'Common Cold', 27: 'Pneumonia',
    28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack', 30: 'Varicose veins',
    31: 'Hypothyroidism', 32: 'Hyperthyroidism', 33: 'Hypoglycemia', 34: 'Osteoarthristis',
    35: 'Arthritis', 36: '(vertigo) Paroymsal  Positional Vertigo', 37: 'Acne',
    38: 'Urinary tract infection', 39: 'Psoriasis', 40: 'Impetigo'
}

# Helper function
def helper(dis):
    try:
        desc = description[description['Disease'] == dis]['Description']
        desc = " ".join([w for w in desc]) if not desc.empty else "No description available"

        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [col for col in pre.values] if not pre.empty else []

        med = medications[medications['Disease'] == dis]['Medication']
        med = [med for med in med.values] if not med.empty else []

        die = diets[diets['Disease'] == dis]['Diet']
        die = [die for die in die.values] if not die.empty else []

        wrkout = workout[workout['disease'] == dis]['workout']
        wrkout = [w for w in wrkout.values] if not wrkout.empty else []

        return desc, pre, med, die, wrkout
    except Exception as e:
        print(f"Error in helper function: {e}")
        return "Error retrieving information", [], [], [], []

# Prediction function
def get_predicted_value(patient_symptoms):
    try:
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1

        prediction = svc.predict([input_vector])[0]
        return diseases_list.get(prediction, "Unknown disease")
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction error"

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print("Received symptoms:", symptoms)

        if not symptoms or symptoms.strip().lower() == "symptoms" or not symptoms.strip():
            message = "Please enter valid symptoms (comma-separated)."
            return render_template('index.html', message=message)

        # Clean and process symptoms
        user_symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms if symptom.strip()]

        # Remove empty symptoms
        user_symptoms = [s for s in user_symptoms if s]

        if not user_symptoms:
            message = "Please enter valid symptoms."
            return render_template('index.html', message=message)

        try:
            predicted_disease = get_predicted_value(user_symptoms)

            if predicted_disease == "Prediction error" or predicted_disease == "Unknown disease":
                message = "Unable to predict disease. Please check your symptoms and try again."
                return render_template('index.html', message=message)

            dis_des, pre, med, rec_diet, wrkout = helper(predicted_disease)

            # Ensure precautions is a list
            my_precautions = []
            if pre:
                if isinstance(pre[0], (list, np.ndarray)):
                    my_precautions = [p for p in pre[0] if pd.notna(p) and p.strip()]
                else:
                    my_precautions = [p for p in pre if pd.notna(p) and str(p).strip()]

            return render_template('index.html',
                                   predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=med,
                                   my_diet=rec_diet,
                                   workout=wrkout)

        except Exception as e:
            print(f"Error in prediction route: {e}")
            return render_template('index.html', message=f"An error occurred. Please try again.")

    return render_template('index.html')

# Other routes
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
