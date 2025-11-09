import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Health Analysis",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - removed heavy gradients
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #667eea;
        text-align: center;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

DATA_FILE = 'patient_records.json'

def save_patient_record(record):
    records = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                records = json.load(f)
        except:
            pass
    records.append(record)
    with open(DATA_FILE, 'w') as f:
        json.dump(records, f)

# Optimized model training with caching
@st.cache_resource(show_spinner=False)
def train_model(df):
    """Train the Random Forest model - CACHED"""
    dataset_df = df.copy()
    
    categorical_columns = ['Gender', 'PhysicalActivity', 'FrequentConsumptionHighCalorieFood', 
                          'FrequentVegetableConsumption']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        dataset_df[col] = le.fit_transform(dataset_df[col])
        label_encoders[col] = le
    
    target_encoder = LabelEncoder()
    dataset_df['Category_encoded'] = target_encoder.fit_transform(dataset_df['Category'])
    
    feature_columns = [
        'Height_m', 'Weight_kg', 'Age', 'Gender', 'PhysicalActivity',
        'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption',
        'BMI', 'Water_Intake_L', 'Sleep_Hours', 'Sleep_Quality_Score',
        'Screen_Time_Hours', 'Steps_Per_Day', 'Protein_Intake_g', 'Stress_Level_Score'
    ]
    
    X = dataset_df[feature_columns]
    y = dataset_df['Category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Reduced trees for faster training
    rf_multi = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    rf_multi.fit(X_train_scaled, y_train)
    
    y_pred = rf_multi.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf_multi, scaler, label_encoders, target_encoder, feature_columns, accuracy

# Cached dataset loading
@st.cache_data(show_spinner=False)
def load_dataset():
    """Load dataset - CACHED"""
    return pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')

# Optimized chart generation
@st.cache_data(show_spinner=False)
def generate_charts(_df):
    """Generate charts - CACHED"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    # 1. Category
    category_counts = _df['Category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726'])
    axes[0, 0].set_title('Category Distribution', fontsize=12, fontweight='bold')
    
    # 2. Gender
    gender_counts = _df['Gender'].value_counts()
    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#667eea', '#764ba2'])
    axes[0, 1].set_title('Gender Distribution', fontsize=12, fontweight='bold')
    
    # 3. Activity
    activity_counts = _df['PhysicalActivity'].value_counts()
    axes[0, 2].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5'])
    axes[0, 2].set_title('Physical Activity', fontsize=12, fontweight='bold')
    
    # 4. High Calorie
    calorie_counts = _df['FrequentConsumptionHighCalorieFood'].value_counts()
    axes[1, 0].pie(calorie_counts.values, labels=['Rarely', 'Frequently'], autopct='%1.1f%%',
                   startangle=90, colors=['#66bb6a', '#ff6b6b'])
    axes[1, 0].set_title('High Calorie Food', fontsize=12, fontweight='bold')
    
    # 5. Vegetables
    veg_counts = _df['FrequentVegetableConsumption'].value_counts()
    axes[1, 1].pie(veg_counts.values, labels=['Regularly', 'Rarely'], autopct='%1.1f%%',
                   startangle=90, colors=['#66bb6a', '#ffa726'])
    axes[1, 1].set_title('Vegetable Consumption', fontsize=12, fontweight='bold')
    
    # 6. Age Groups
    age_bins = [0, 20, 30, 40, 50, 100]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '50+']
    age_group = pd.cut(_df['Age'], bins=age_bins, labels=age_labels)
    age_counts = age_group.value_counts()
    axes[1, 2].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5', '#ab47bc'])
    axes[1, 2].set_title('Age Groups', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_suggestions(category, user_data):
    suggestions = []
    action_plan = []
    
    if category == 'Underweight':
        suggestions.append("Increase calorie intake by 300-500 calories daily")
        suggestions.append("Focus on protein-rich meals")
        action_plan.extend([
            "Week 1-2: Increase protein to 70-80g daily",
            "Week 3-4: Add strength training 2-3x per week"
        ])
    elif category in ['Overweight', 'Obese']:
        suggestions.append("Increase daily steps to 10,000")
        suggestions.append("Reduce high-calorie foods")
        action_plan.extend([
            "Week 1-2: Walk 30 min daily",
            "Week 3-4: Add meal planning"
        ])
    else:
        suggestions.append("Maintain current healthy habits")
        action_plan.append("Monitor weight monthly")
    
    if user_data['sleep'] < 7:
        suggestions.append(f"Increase sleep to 7-9 hours")
    if user_data['water'] < 2:
        suggestions.append("Increase water to 2-3L daily")
    if user_data['stress'] > 6:
        suggestions.append("Practice stress management")
    
    return suggestions, action_plan

def main():
    st.markdown("<h1>🏃‍♂️ Health Analysis System</h1>", unsafe_allow_html=True)
    
    # Load dataset with progress
    try:
        df = load_dataset()
        
        # Train model only once
        if not st.session_state.model_trained:
            with st.spinner('⏳ Loading AI model...'):
                model, scaler, label_encoders, target_encoder, feature_columns, accuracy = train_model(df)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.label_encoders = label_encoders
                st.session_state.target_encoder = target_encoder
                st.session_state.feature_columns = feature_columns
                st.session_state.accuracy = accuracy
                st.session_state.df = df
                st.session_state.model_trained = True
            st.success(f'✅ Model Ready! (Accuracy: {accuracy:.2%})')
            
    except FileNotFoundError:
        st.error("❌ Dataset file not found!")
        st.info("Please upload: augmented_obesity_lifestyle_dataset (1).csv")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_csv('augmented_obesity_lifestyle_dataset (1).csv', index=False)
            st.success("✅ Uploaded! Refresh the page.")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Visualization", "📋 Dataset"])
    
    # Tab 1: Prediction
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Patient Info")
            name = st.text_input("Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            date = st.date_input("Date", datetime.now())
            
            st.subheader("📊 Metrics")
            height = st.number_input("Height (m)", 1.0, 2.5, 1.75, 0.01)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 68.0, 0.1)
            age = st.number_input("Age", 10, 100, 28)
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            activity = st.selectbox("Activity", ["Sedentary", "Light", "Moderate", "High"])
            steps = st.number_input("Daily Steps", 0, 50000, 8500, 100)
        
        with col2:
            st.subheader("🍽️ Nutrition")
            protein = st.number_input("Protein (g/day)", 0, 300, 75)
            water = st.number_input("Water (L/day)", 0.0, 10.0, 2.5, 0.1)
            highCalorie = st.selectbox("High-Calorie Food", ["no", "yes"])
            vegetables = st.selectbox("Vegetables", ["yes", "no"])
            
            st.subheader("😴 Wellness")
            sleep = st.number_input("Sleep (hours)", 0.0, 24.0, 7.5, 0.5)
            sleepQuality = st.slider("Sleep Quality", 1, 10, 7)
            stress = st.slider("Stress Level", 1, 10, 4)
            screenTime = st.number_input("Screen Time (h)", 0.0, 24.0, 5.0, 0.5)
        
        if st.button("🔍 Analyze", use_container_width=True):
            if not name or not email or not phone:
                st.warning("⚠️ Fill all patient info!")
            else:
                # Calculate
                bmi = weight / (height ** 2)
                
                # Prepare input
                user_input = []
                for col in st.session_state.feature_columns:
                    if col == 'BMI':
                        user_input.append(bmi)
                    elif col == 'Height_m':
                        user_input.append(height)
                    elif col == 'Weight_kg':
                        user_input.append(weight)
                    elif col == 'Age':
                        user_input.append(age)
                    elif col == 'Gender':
                        user_input.append(st.session_state.label_encoders['Gender'].transform([gender])[0])
                    elif col == 'PhysicalActivity':
                        user_input.append(st.session_state.label_encoders['PhysicalActivity'].transform([activity])[0])
                    elif col == 'FrequentConsumptionHighCalorieFood':
                        user_input.append(st.session_state.label_encoders['FrequentConsumptionHighCalorieFood'].transform([highCalorie])[0])
                    elif col == 'FrequentVegetableConsumption':
                        user_input.append(st.session_state.label_encoders['FrequentVegetableConsumption'].transform([vegetables])[0])
                    elif col == 'Water_Intake_L':
                        user_input.append(water)
                    elif col == 'Sleep_Hours':
                        user_input.append(sleep)
                    elif col == 'Sleep_Quality_Score':
                        user_input.append(sleepQuality)
                    elif col == 'Screen_Time_Hours':
                        user_input.append(screenTime)
                    elif col == 'Steps_Per_Day':
                        user_input.append(steps)
                    elif col == 'Protein_Intake_g':
                        user_input.append(protein)
                    elif col == 'Stress_Level_Score':
                        user_input.append(stress)
                
                # Predict
                user_input_scaled = st.session_state.scaler.transform([user_input])
                prediction = st.session_state.model.predict(user_input_scaled)[0]
                probabilities = st.session_state.model.predict_proba(user_input_scaled)[0]
                
                category = st.session_state.target_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction]
                
                # BMI Category
                if bmi < 18.5:
                    bmi_category = "Underweight"
                elif 18.5 <= bmi < 25:
                    bmi_category = "Normal"
                elif 25 <= bmi < 30:
                    bmi_category = "Overweight"
                else:
                    bmi_category = "Obese"
                
                # Results
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                st.markdown(f"### 📊 Report for {name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Category", category)
                with col2:
                    st.metric("💪 Confidence", f"{confidence*100:.1f}%")
                with col3:
                    st.metric("📊 BMI", f"{bmi:.1f} ({bmi_category})")
                
                # Feature importance
                importances = st.session_state.model.feature_importances_
                top_indices = np.argsort(importances)[-3:][::-1]
                
                with st.expander("🔍 Top Factors", expanded=True):
                    for idx in top_indices:
                        feat_name = st.session_state.feature_columns[idx]
                        feat_val = user_input[idx]
                        feat_imp = importances[idx]
                        st.write(f"**{feat_name}:** {feat_val:.2f if isinstance(feat_val, float) else feat_val} ({feat_imp*100:.1f}%)")
                
                # Suggestions
                user_data = {'bmi': bmi, 'sleep': sleep, 'water': water, 'stress': stress, 
                           'screenTime': screenTime, 'steps': steps, 'protein': protein}
                suggestions, action_plan = get_suggestions(category, user_data)
                
                with st.expander("💡 Recommendations", expanded=True):
                    for sug in suggestions:
                        st.write(f"• {sug}")
                
                if action_plan:
                    with st.expander("📅 Action Plan"):
                        for plan in action_plan:
                            st.write(f"• {plan}")
                
                # Save
                record = {
                    'name': name, 'email': email, 'phone': phone,
                    'date': str(date), 'category': category, 'bmi': bmi,
                    'confidence': float(confidence), 'timestamp': datetime.now().isoformat()
                }
                save_patient_record(record)
    
    # Tab 2: Visualization
    with tab2:
        st.subheader("📊 Dataset Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        df = st.session_state.df
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        with col3:
            st.metric("Avg BMI", f"{df['BMI'].mean():.1f}")
        with col4:
            gender_counts = df['Gender'].value_counts()
            male_pct = (gender_counts.get('Male', 0) / len(df) * 100)
            st.metric("Male %", f"{male_pct:.1f}%")
        with col5:
            female_pct = (gender_counts.get('Female', 0) / len(df) * 100)
            st.metric("Female %", f"{female_pct:.1f}%")
        
        st.markdown("---")
        
        with st.spinner('Generating charts...'):
            fig = generate_charts(df)
            st.pyplot(fig)
    
    # Tab 3: Dataset
    with tab3:
        st.subheader("📋 Training Data")
        
        # Show sample
        st.dataframe(st.session_state.df.head(100), use_container_width=True)
        st.info(f"Showing 100 of {len(st.session_state.df)} records")
        
        # Download
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Dataset",
            csv,
            "dataset.csv",
            "text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
