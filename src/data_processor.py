import pandas as pd
import numpy as np
from config.config import FEATURE_MAPS

class DataProcessor:
    def __init__(self):
        self.feature_maps = FEATURE_MAPS
        self.unified_datasets = {}
        self.combined_data = None

    def load_datasets(self, file_paths):
        datasets = {}
        print("Loading datasets...")
        for name, path in file_paths.items():
            try:
                if name == 'additional_train':
                    X_path = './dataset/X_train_processed.csv'
                    y_path = './dataset/y_train_processed.csv'
                    X_df = pd.read_csv(X_path)
                    y_df = pd.read_csv(y_path)
                    df = X_df.copy()
                    df['Burn Rate'] = y_df['Burn Rate']
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns (merged from X and y)")
                else:
                    df = pd.read_csv(path)
                    datasets[name] = df
                    print(f"✓ Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        return datasets

    def create_unified_features(self, df, dataset_name):
        unified_df = pd.DataFrame()

        if dataset_name not in self.feature_maps:
            print(f"Warning: No mapping for {dataset_name}")
            return None

        feature_map = self.feature_maps[dataset_name]

        common_features = ['stress_level', 'workload', 'sleep_quality', 
                          'support', 'screen_time', 'social_media', 'work_setup']

        for feat in common_features:
            if feat in feature_map and feature_map[feat] in df.columns:
                unified_df[feat] = df[feature_map[feat]]
            else:
                unified_df[feat] = np.nan

        if 'target_col' in feature_map and feature_map['target_col'] in df.columns:
            target_transform = feature_map.get('target_transform', lambda x: x)
            unified_df['burnout_target'] = target_transform(df[feature_map['target_col']])

        unified_df['dataset_source'] = dataset_name

        if 'work_setup' in unified_df.columns:
            unified_df['work_setup'] = unified_df['work_setup'].replace({'Yes': 1, 'No': 0, 'Hybrid': 0.5})
            unified_df['work_setup'] = pd.to_numeric(unified_df['work_setup'], errors='coerce').fillna(0)

        if 'support' in unified_df.columns:
            unified_df['support'] = unified_df['support'].replace({'Yes': 1, 'No': 0})
            unified_df['support'] = pd.to_numeric(unified_df['support'], errors='coerce')

        if dataset_name == 'screen_time' and 'stress_level' in unified_df.columns:
            unified_df['stress_level'] = unified_df['stress_level'].replace({'Low': 3, 'Medium': 6, 'High': 9})
            unified_df['stress_level'] = pd.to_numeric(unified_df['stress_level'], errors='coerce')

        return unified_df

    def merge_datasets(self, datasets):
        print("\nCreating unified feature schema...")
        unified_list = []

        for name, df in datasets.items():
            unified = self.create_unified_features(df, name)
            if unified is not None:
                self.unified_datasets[name] = unified
                unified_list.append(unified)
                print(f"✓ Unified {name}: {unified.shape[0]} samples")

        print("\nMerging datasets...")
        self.combined_data = pd.concat(unified_list, ignore_index=True)

        print(f"\n✓ Combined dataset created: {self.combined_data.shape[0]} total samples")
        print(f"  Features: {self.combined_data.shape[1]}")
        print(f"  Missing values:\n{self.combined_data.isnull().sum()}")

        return self.combined_data

    def preprocess_combined_data(self):
        print("\nPreprocessing combined data...")

        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('burnout_target', errors='ignore')

        for col in numeric_cols:
            if self.combined_data[col].isnull().any():
                median_val = self.combined_data[col].median()
                self.combined_data[col] = self.combined_data[col].fillna(median_val)

        print("Creating interaction features...")

        if 'stress_level' in self.combined_data.columns and 'sleep_quality' in self.combined_data.columns:
            self.combined_data['stress_sleep_ratio'] = (
                self.combined_data['stress_level'] / (self.combined_data['sleep_quality'] + 1)
            )

        if 'screen_time' in self.combined_data.columns and 'social_media' in self.combined_data.columns:
            self.combined_data['digital_overload'] = (
                self.combined_data['screen_time'] * self.combined_data['social_media']
            ).fillna(0)

        if 'workload' in self.combined_data.columns and 'stress_level' in self.combined_data.columns:
            self.combined_data['workload_stress'] = (
                self.combined_data['workload'] * self.combined_data['stress_level']
            ).fillna(0)

        self.combined_data = pd.get_dummies(
            self.combined_data, 
            columns=['dataset_source'], 
            drop_first=False,
            dtype=int
        )

        print("✓ Preprocessing complete")
        print(f"  Final shape: {self.combined_data.shape}")

        return self.combined_data

    def handle_missing_values(self):
        print("\n" + "="*70)
        print("STEP 1: HANDLING MISSING VALUES (IMPROVED)")
        print("="*70)

        print("\n📊 Missing Values Before Cleaning:")
        missing_summary = self.combined_data.isnull().sum()
        missing_pct = (missing_summary / len(self.combined_data)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing_Count': missing_summary.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )

        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))

        print("\n💡 Strategy: Filling ALL columns (not dropping any)")

        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('burnout_target', errors='ignore')

        for col in numeric_cols:
            if self.combined_data[col].isnull().any():
                missing_pct = (self.combined_data[col].isnull().sum() / len(self.combined_data)) * 100

                if missing_pct > 50:
                    print(f"\n  📍 {col} ({missing_pct:.1f}% missing):")

                    indicator_col = f'{col}_was_missing'
                    self.combined_data[indicator_col] = self.combined_data[col].isnull().astype(int)
                    print(f"     ✓ Created indicator: {indicator_col}")

                    if 'burnout_target' in self.combined_data.columns:
                        self.combined_data[col] = self.combined_data.groupby('burnout_target')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                        print(f"     ✓ Filled with group-specific median")
                    else:
                        median_val = self.combined_data[col].median()
                        self.combined_data[col] = self.combined_data[col].fillna(median_val)
                        print(f"     ✓ Filled with overall median: {median_val:.2f}")
                else:
                    median_val = self.combined_data[col].median()
                    missing_count = self.combined_data[col].isnull().sum()
                    self.combined_data[col] = self.combined_data[col].fillna(median_val)
                    print(f"  ✓ {col}: filled {missing_count} values with {median_val:.2f}")

        categorical_cols = self.combined_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.combined_data[col].isnull().any():
                mode_val = self.combined_data[col].mode()[0] if len(self.combined_data[col].mode()) > 0 else 'Unknown'
                missing_count = self.combined_data[col].isnull().sum()
                self.combined_data[col] = self.combined_data[col].fillna(mode_val)
                print(f"  ✓ {col}: filled {missing_count} values with '{mode_val}'")

        print("\n🎯 Handling target variable...")
        if 'burnout_target' in self.combined_data.columns:
            missing_target = self.combined_data['burnout_target'].isnull().sum()
            if missing_target > 0:
                print(f"  ⚠️  Removing {missing_target} rows with missing target")
                self.combined_data.dropna(subset=['burnout_target'], inplace=True)
            else:
                print("  ✓ No missing target values")

        print("\n✅ Missing Values After Cleaning:")
        final_missing = self.combined_data.isnull().sum().sum()
        print(f"  Total missing values: {final_missing}")

        if final_missing == 0:
            print("  ✓ All missing values handled!")

        print(f"\n📊 Final dataset shape: {self.combined_data.shape}")
        print(f"📊 Features gained from missing indicators: {len([c for c in self.combined_data.columns if '_was_missing' in c])}")
        print("="*70)

        return self.combined_data
