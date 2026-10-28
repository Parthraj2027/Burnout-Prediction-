from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATASET_PATHS = {
    'employee_burnout': './dataset/train.csv',
    'hackerearth_burnout': './dataset/train.csv',
    'workplace_mental': './dataset/mental_health_workplace_survey.csv',
    'student_resilience': './dataset/mental_health_dataset.csv',
    'student_stress': './dataset/StressLevelDataset.csv',
    'tech_usage': './dataset/digital_habits_vs_mental_health.csv',
    'screen_time': './dataset/mental_health_and_technology_usage_2024.csv',
    'healthcare_mental': './dataset/Healthcare Workforce Mental Health Dataset.csv',
    'additional_train': './dataset/X_train_processed.csv'
}

MODEL_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 300,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.3,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss'
}

PROCESSING_PARAMS = {
    'test_size': 0.15,
    'validation_size': 0.15,
    'random_state': 42,
    'missing_threshold': 50,
    'correlation_threshold': 0.8
}

SMOTE_PARAMS = {
    'random_state': 42
}

OUTPUT_PATHS = {
    'model': './models/burnout_combined_model.pkl',
    'correlation_full': './models/correlation_matrix_full.png',
    'correlation_focused': './models/correlation_matrix_focused.png'
}

PLOT_SETTINGS = {
    'full_heatmap_size': (16, 14),
    'focused_heatmap_size': (12, 10),
    'dpi': 300,
    'cmap': 'RdBu_r'
}

FEATURE_MAPS = {
    'employee_burnout': {
        'stress_level': 'Mental Fatigue Score',
        'workload': 'Resource Allocation',
        'work_setup': 'WFH Setup Available',
        'company_type': 'Company Type',
        'target_col': 'Burn Rate',
        'target_transform': lambda x: (x > 0.5).astype(int)
    },
    'hackerearth_burnout': {
        'stress_level': 'Mental Fatigue Score',
        'workload': 'Resource Allocation',
        'work_setup': 'WFH Setup Available',
        'target_col': 'Burn Rate',
        'target_transform': lambda x: (x > 0.5).astype(int)
    },
    'workplace_mental': {
        'stress_level': 'StressLevel',
        'work_setup': 'RemoteWork',
        'support': 'HasMentalHealthSupport',
        'target_col': 'BurnoutRisk',
        'target_transform': lambda x: x.astype(int)
    },
    'student_resilience': {
        'stress_level': 'Anxiety_Score',
        'workload': 'Stress_Level',
        'sleep_quality': 'Sleep_Hours',
        'support': 'Sentiment_Score',
        'target_col': 'Mental_Health_Status',
        'target_transform': lambda x: (x >= 2).astype(int)
    },
    'student_stress': {
        'stress_level': 'anxiety_level',
        'workload': 'study_load',
        'sleep_quality': 'sleep_quality',
        'support': 'social_support',
        'target_col': 'stress_level',
        'target_transform': lambda x: (x >= 1).astype(int)
    },
    'tech_usage': {
        'stress_level': 'stress_level',
        'screen_time': 'screen_time_hours',
        'social_media': 'social_media_platforms_used',
        'sleep_quality': 'sleep_hours',
        'target_col': 'mood_score',
        'target_transform': lambda x: (x < 5).astype(int)
    },
    'screen_time': {
        'stress_level': 'Stress_Level',
        'screen_time': 'Screen_Time_Hours',
        'social_media': 'Social_Media_Usage_Hours',
        'sleep_quality': 'Sleep_Hours',
        'target_col': 'Mental_Health_Status',
        'target_transform': lambda x: x.isin(['Poor', 'Fair']).astype(int)
    },
    'healthcare_mental': {
        'stress_level': 'Stress Level',
        'support': 'Access to EAPs',
        'target_col': 'Burnout Frequency',
        'target_transform': lambda x: (x == 'Often').astype(int)
    },
    'additional_train': {
        'stress_level': 'Mental Fatigue Score',
        'workload': 'Resource Allocation',
        'work_setup': 'WFH Setup Available_Yes',
        'company_type': 'Company Type_Service',
        'target_col': 'Burn Rate',
        'target_transform': lambda x: (x > 0.5).astype(int)
    }
}
