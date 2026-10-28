import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
"""
Burnout Prediction - User Interface
Interactive system to check burnout risk for individuals
Run this AFTER training the model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BurnoutChecker:
    def __init__(self, model_path='burnout_combined_model.pkl'):
        print("="*60)
        print("🔥 BURNOUT RISK CHECKER")
        print("="*60)
        print("\nLoading trained model...")

        possible_paths = [
            model_path,
            'burnout_combined_model.pkl',
            './models/burnout_combined_model.pkl',
            './models/single_model_xgboost.pkl',
            './models/hybrid_ensemble.pkl',
        ]
        loaded = False
        last_error = None
        for path in possible_paths:
            if Path(path).exists():
                try:
                    model_data = joblib.load(path)
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model', model_data.get('ensemble'))
                        self.scaler = model_data.get('scaler', None)
                        self.feature_names = model_data.get('feature_names', None)
                    else:
                        self.model = model_data
                        self.scaler = None
                        self.feature_names = None
                    if self.model is not None:
                        print(f"✓ Model loaded successfully from '{path}'!\n")
                        loaded = True
                        break
                except Exception as e:
                    last_error = e
                    continue
        if not loaded:
            print(f"\n❌ Error: Model file '{model_path}' not found in candidate paths!")
            print("Please run the training pipeline first (e.g. 'python run_pipeline.py').")
            if last_error:
                raise last_error
            raise FileNotFoundError(f"No model found at {model_path}")

    def get_user_input(self):
        print("="*60)
        print("📝 PLEASE ANSWER THE FOLLOWING QUESTIONS")
        print("="*60)
        print("\nNote: Answer on a scale of 0-10 unless specified otherwise\n")

        features = {}

        print("1️⃣  STRESS LEVEL")
        print("   How stressed do you feel on a daily basis?")
        print("   (0 = Not stressed at all, 10 = Extremely stressed)")
        features['stress_level'] = self.get_numeric_input("   Your answer: ", 0, 10)

        print("\n2️⃣  WORKLOAD")
        print("   How would you rate your current workload/study load?")
        print("   (0 = Very light, 10 = Extremely heavy)")
        features['workload'] = self.get_numeric_input("   Your answer: ", 0, 10)

        print("\n3️⃣  SLEEP QUALITY")
        print("   How would you rate your sleep quality?")
        print("   (0 = Very poor, 10 = Excellent)")
        features['sleep_quality'] = self.get_numeric_input("   Your answer: ", 0, 10)

        print("\n4️⃣  SUPPORT SYSTEM")
        print("   How supported do you feel by family/friends/colleagues?")
        print("   (0 = No support at all, 10 = Excellent support)")
        features['support'] = self.get_numeric_input("   Your answer: ", 0, 10)

        print("\n5️⃣  SCREEN TIME")
        print("   How many hours per day do you spend on screens (work + leisure)?")
        print("   (Enter hours, e.g., 8.5)")
        features['screen_time'] = self.get_numeric_input("   Your answer: ", 0, 24)

        print("\n6️⃣  SOCIAL MEDIA USAGE")
        print("   How many hours per day do you spend on social media?")
        print("   (Enter hours, e.g., 3.0)")
        features['social_media'] = self.get_numeric_input("   Your answer: ", 0, 24)

        print("\n7️⃣  WORK/STUDY ENVIRONMENT")
        print("   Do you work/study from home?")
        print("   (1 = Yes, 0 = No)")
        features['work_setup'] = self.get_numeric_input("   Your answer (0 or 1): ", 0, 1)

        return features

    def get_numeric_input(self, prompt, min_val, max_val):
        while True:
            try:
                value = float(input(prompt))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"   ⚠️  Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("   ⚠️  Please enter a valid number")

    def predict_burnout(self, features):
        df = pd.DataFrame([features])

        if self.feature_names:
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[self.feature_names]

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0, 1]

        return prediction, probability

    def show_results(self, prediction, probability, features):
        print("\n" + "="*60)
        print("🔍 ANALYSIS RESULTS")
        print("="*60)

        print(f"\n📊 BURNOUT RISK: {probability*100:.1f}%")

        if prediction == 1:
            print("🔴 STATUS: HIGH RISK - Burnout detected!")
            risk_level = "HIGH"
            emoji = "🔴"
        else:
            if probability > 0.7:
                print("🟡 STATUS: MODERATE RISK - Watch out!")
                risk_level = "MODERATE"
                emoji = "🟡"
            elif probability > 0.4:
                print("🟢 STATUS: LOW RISK - You're doing okay")
                risk_level = "LOW"
                emoji = "🟢"
            else:
                print("🟢 STATUS: VERY LOW RISK - Great job!")
                risk_level = "VERY LOW"
                emoji = "🟢"

        print(f"\n📈 RISK BREAKDOWN:")
        print(f"   {emoji} Overall Risk Score: {probability*100:.1f}%")
        print(f"   {'🔴' if probability > 0.7 else '🟢'} Probability of Burnout: {probability:.3f}")
        print(f"   {'🟢' if probability < 0.3 else '🔴'} Probability of No Burnout: {1-probability:.3f}")

        print(f"\n🔑 YOUR KEY FACTORS:")
        self.show_factor_analysis(features)

        print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        self.show_recommendations(features, risk_level)

        print("\n" + "="*60)

    def show_factor_analysis(self, features):
        factors = []

        if features.get('stress_level', 0) >= 7:
            factors.append("⚠️  High stress levels detected")
        elif features.get('stress_level', 0) <= 3:
            factors.append("✓ Good stress management")

        if features.get('workload', 0) >= 8:
            factors.append("⚠️  Heavy workload")
        elif features.get('workload', 0) <= 4:
            factors.append("✓ Manageable workload")

        if features.get('sleep_quality', 0) <= 4:
            factors.append("⚠️  Poor sleep quality")
        elif features.get('sleep_quality', 0) >= 7:
            factors.append("✓ Good sleep quality")

        if features.get('support', 0) <= 4:
            factors.append("⚠️  Limited support system")
        elif features.get('support', 0) >= 7:
            factors.append("✓ Strong support system")

        if features.get('screen_time', 0) >= 12:
            factors.append("⚠️  Excessive screen time")
        elif features.get('screen_time', 0) <= 6:
            factors.append("✓ Healthy screen time")

        if features.get('social_media', 0) >= 4:
            factors.append("⚠️  High social media usage")
        elif features.get('social_media', 0) <= 2:
            factors.append("✓ Limited social media usage")

        for factor in factors:
            print(f"   {factor}")

        if not factors:
            print("   ✓ All factors within normal range")

    def show_recommendations(self, features, risk_level):
        recommendations = []

        if features.get('stress_level', 0) >= 7:
            recommendations.append(
                "🧘 STRESS: Practice mindfulness or meditation for 10 minutes daily"
            )
            recommendations.append(
                "🧘 STRESS: Consider talking to a counselor or therapist"
            )

        if features.get('sleep_quality', 0) <= 4:
            recommendations.append(
                "😴 SLEEP: Establish a consistent sleep schedule (7-8 hours)"
            )
            recommendations.append(
                "😴 SLEEP: Avoid screens 1 hour before bedtime"
            )

        if features.get('workload', 0) >= 8:
            recommendations.append(
                "📋 WORKLOAD: Prioritize tasks and delegate when possible"
            )
            recommendations.append(
                "📋 WORKLOAD: Take regular breaks (5-10 mins every hour)"
            )

        if features.get('support', 0) <= 4:
            recommendations.append(
                "👥 SUPPORT: Reach out to friends or family regularly"
            )
            recommendations.append(
                "👥 SUPPORT: Join a support group or community"
            )

        if features.get('screen_time', 0) >= 12:
            recommendations.append(
                "📱 SCREEN TIME: Use the 20-20-20 rule (every 20 mins, look 20 feet away for 20 seconds)"
            )
            recommendations.append(
                "📱 SCREEN TIME: Set screen time limits and take tech-free breaks"
            )

        if features.get('social_media', 0) >= 4:
            recommendations.append(
                "📲 SOCIAL MEDIA: Limit usage to 1-2 hours per day"
            )
            recommendations.append(
                "📲 SOCIAL MEDIA: Turn off non-essential notifications"
            )

        if risk_level == "HIGH":
            recommendations.append(
                "⚠️  URGENT: Consider consulting a mental health professional"
            )
            recommendations.append(
                "⚠️  URGENT: Take time off work/study if possible"
            )

        recommendations.append(
            "💪 EXERCISE: Get 30 minutes of physical activity daily"
        )
        recommendations.append(
            "🥗 NUTRITION: Maintain a balanced diet and stay hydrated"
        )
        recommendations.append(
            "🎯 GOALS: Set realistic daily goals and celebrate small wins"
        )

        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    def run_interactive_mode(self):
        while True:
            try:
                features = self.get_user_input()

                print("\n⏳ Analyzing your responses...")
                prediction, probability = self.predict_burnout(features)

                self.show_results(prediction, probability, features)

                print("\n" + "="*60)
                again = input("\n❓ Would you like to check for another person? (yes/no): ")
                if again.lower() not in ['yes', 'y']:
                    print("\n👋 Thank you for using Burnout Risk Checker!")
                    print("Stay healthy and take care of yourself! 💚")
                    print("="*60)
                    break

                print("\n")

            except KeyboardInterrupt:
                print("\n\n👋 Exiting... Take care!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                again = input("\nWould you like to try again? (yes/no): ")
                if again.lower() not in ['yes', 'y']:
                    break

def quick_check():
    print("="*60)
    print("🚀 QUICK DEMO MODE")
    print("="*60)

    checker = BurnoutChecker()

    profiles = [
        {
            'name': 'High Risk Employee',
            'features': {
                'stress_level': 9.0,
                'workload': 9.5,
                'sleep_quality': 2.0,
                'support': 3.0,
                'screen_time': 14.0,
                'social_media': 5.0,
                'work_setup': 1
            }
        },
        {
            'name': 'Healthy Individual',
            'features': {
                'stress_level': 3.0,
                'workload': 5.0,
                'sleep_quality': 8.0,
                'support': 8.0,
                'screen_time': 6.0,
                'social_media': 2.0,
                'work_setup': 0
            }
        }
    ]

    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"👤 PROFILE: {profile['name']}")
        print(f"{'='*60}")

        prediction, probability = checker.predict_burnout(profile['features'])
        checker.show_results(prediction, probability, profile['features'])

        input("\nPress Enter to continue...")

def main():
    print("\n")
    print("="*60)
    print("🔥 BURNOUT RISK PREDICTION SYSTEM")
    print("="*60)
    print("\nChoose an option:")
    print("1. Interactive Mode (Answer questions)")
    print("2. Quick Demo (See sample predictions)")
    print("3. Exit")

    choice = input("\nEnter your choice (1/2/3): ")

    if choice == '1':
        try:
            checker = BurnoutChecker()
            checker.run_interactive_mode()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure you've trained the model first!")

    elif choice == '2':
        try:
            quick_check()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure you've trained the model first!")

    elif choice == '3':
        print("\n👋 Goodbye!")

    else:
        print("\n❌ Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()
