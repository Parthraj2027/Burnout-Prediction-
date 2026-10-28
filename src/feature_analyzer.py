import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.config import PLOT_SETTINGS, OUTPUT_PATHS

class FeatureAnalyzer:
    def __init__(self, data):
        self.data = data
        self.correlation_matrix = None

    def analyze_correlations(self, save_plot=True):
        print("\n" + "="*70)
        print("STEP 2: CORRELATION ANALYSIS")
        print("="*70)

        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            print("⚠️  Not enough numeric columns for correlation analysis")
            return None

        print(f"\n📊 Analyzing {numeric_data.shape[1]} numeric features...")

        self.correlation_matrix = numeric_data.corr()

        if 'burnout_target' in self.correlation_matrix.columns:
            print("\n🎯 TOP CORRELATIONS WITH BURNOUT:")
            target_corr = self.correlation_matrix['burnout_target'].sort_values(ascending=False)
            target_corr = target_corr[target_corr.index != 'burnout_target']

            print("\nPositive correlations (increase burnout risk):")
            positive = target_corr[target_corr > 0].head(10)
            for feat, corr in positive.items():
                print(f"  {feat:30s}: {corr:+.4f}")

            print("\nNegative correlations (decrease burnout risk):")
            negative = target_corr[target_corr < 0].head(10)
            for feat, corr in negative.items():
                print(f"  {feat:30s}: {corr:+.4f}")

        print("\n⚠️  HIGHLY CORRELATED FEATURES (potential multicollinearity):")
        high_corr_pairs = []

        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                if abs(self.correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'Feature 1': self.correlation_matrix.columns[i],
                        'Feature 2': self.correlation_matrix.columns[j],
                        'Correlation': self.correlation_matrix.iloc[i, j]
                    })

        if high_corr_pairs:
            for pair in high_corr_pairs:
                print(f"  {pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.4f}")
            print(f"\n  💡 Consider removing one feature from each pair")
        else:
            print("  ✓ No highly correlated feature pairs found")

        print("\n📈 Creating correlation heatmap...")

        plt.figure(figsize=PLOT_SETTINGS['full_heatmap_size'])

        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))

        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=False,
            cmap=PLOT_SETTINGS['cmap'],
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1
        )

        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        if save_plot:
            Path(OUTPUT_PATHS['correlation_full']).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(OUTPUT_PATHS['correlation_full'], dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
            print(f"  ✓ Saved: {OUTPUT_PATHS['correlation_full']}")

        plt.close()

        if 'burnout_target' in self.correlation_matrix.columns:
            print("\n📈 Creating focused correlation heatmap (top features)...")

            target_corr = self.correlation_matrix['burnout_target'].abs().sort_values(ascending=False)
            top_features = target_corr.head(16).index.tolist()

            focused_corr = self.correlation_matrix.loc[top_features, top_features]

            plt.figure(figsize=PLOT_SETTINGS['focused_heatmap_size'])
            sns.heatmap(
                focused_corr,
                annot=True,
                fmt='.2f',
                cmap=PLOT_SETTINGS['cmap'],
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1
            )

            plt.title('Top Features - Correlation with Burnout Target', 
                      fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()

            if save_plot:
                Path(OUTPUT_PATHS['correlation_focused']).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(OUTPUT_PATHS['correlation_focused'], dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
                print(f"  ✓ Saved: {OUTPUT_PATHS['correlation_focused']}")

            plt.close()

        print("\n" + "="*70)

        return self.correlation_matrix
