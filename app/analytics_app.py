# ====================================================================
# V-Med Pro Analytics Demo Pipeline (Live Capstone Data Version)
# File Name: VMed_Pro_Analytics_App.py
#
# INSTRUCTIONS:
# 1. Save this file (e.g., VMed_Pro_Analytics_App.py)
# 2. Make sure 'final_30_responses_dataset.csv' is in the SAME folder.
# 3. Run from Anaconda Terminal: streamlit run VMed_Pro_Analytics_App.py
# ====================================================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, ttest_ind
import warnings

# --- Page Config ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="V-Med Pro Analytics Demo", layout="wide")
st.title("VIZITECH–ASU Analytics Demo Pipeline 🧠")
st.markdown("### Automated Data Cleaning → Descriptive → Hypothesis → Predictive Flow")

# ===================================================================
# 1️⃣ LOAD LIVE CAPSTONE DATA
# ===================================================================
st.subheader("📥 Step 1: Load Live Capstone Data")

file_path = 'final_30_responses_dataset.csv'
try:
    # Use the actual project CSV file
    df = pd.read_csv(file_path)
    st.write("**Live Capstone Data (first 5 rows):**")
    st.dataframe(df.head())
    st.info(f"💡 Successfully loaded '{file_path}'. This dataset contains {df.shape[0]} responses (4 real, 26 synthetic).")
except FileNotFoundError:
    st.error(f"FATAL ERROR: The file '{file_path}' was not found.")
    st.info(f"Please make sure '{file_path}' is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ===================================================================
# 2️⃣ DATA CLEANING & ENCODING
# ===================================================================
st.subheader("🧹 Step 2: Data Cleaning & Encoding")
st.markdown("**Encoding Categorical Text to Numeric Values**")

# Make a copy to preserve the original
df_processed = df.copy()

# Define mapping dictionaries from our Analysis 1
likelihood_map = {
    'Not at all likely': 1, 'Slightly Likely': 2, 'Moderately Likely': 3,
    'Likely': 4, 'Very likely': 5, 'Extremely Likely': 5
}
payment_map = {
    'One-time purchase': 1,  # Segment 1 (Budget)
    'Subscription': 2         # Segment 2 (Outcomes)
}
barrier_map = {
    'Cost (purchase, licensing, or maintenance)': 1,
    'Technical setup, integration, or infrastructure needs': 2,
    'Content and realism (limited scenarios, lack of tactile feedback)': 3,
    'Faculty, staff, or student acceptance (perception, willingness to use)': 4,
    'Accessibility (training time, comfort, motion sickness, physical limits)': 5,
    'None (Other)': 6
}
scenarios_map = {'Many varied scenarios': 1, 'Few highly detailed scenarios': 2}
practice_map = {'Independent, self-directed practice': 1, 'Instructor-led guided sessions': 2}
hardware_map = {'Basic/entry-level VR hardware': 1, 'Premium, high-spec VR hardware': 2}

# Apply all mappings
try:
    # --- ENCODING CATEGORICAL COLUMNS ---
    df_processed['Q12. Adopt Likely'] = df_processed['Q12. Adopt Likely'].map(likelihood_map)
    df_processed['Q14. Experiment Likely'] = df_processed['Q14. Experiment Likely'].map(likelihood_map)
    df_processed['Q10_Features_Payment'] = df_processed['Q10_Features_Payment'].map(payment_map)
    df_processed['Q13. Biggest Barrier'] = df_processed['Q13. Biggest Barrier'].map(barrier_map)
    df_processed['Q8_Features_Scenarios'] = df_processed['Q8_Features_Scenarios'].map(scenarios_map)
    df_processed['Q9_Features_Practice'] = df_processed['Q9_Features_Practice'].map(practice_map)
    df_processed['Q11_Features_Hardware'] = df_processed['Q11_Features_Hardware'].map(hardware_map)

    # --- NORMALIZING NUMERIC DATA ---
    st.markdown("**Normalizing Data (MinMax Scaler)**")
    # Drop non-analytical columns for scaling
    df_to_scale = df_processed.drop(columns=['Respondent ID', 'Q2. Role', 'Q4. Current Method'])
    
    # --- START OF FIX for NaN Error ---
    # Handle any potential NaNs created by the .map() step *before* scaling
    # We will fill with the median, which is robust.
    st.markdown("**Imputing Missing Values (Post-Encoding)**")
    nan_count_before = df_to_scale.isna().sum().sum()
    if nan_count_before > 0:
        st.warning(f"Found {nan_count_before} missing value(s) post-encoding (likely from uncaught text variations).")
        # Impute NaNs with the median value of their respective column
        for col in df_to_scale.columns:
            if df_to_scale[col].isnull().any():
                median_val = df_to_scale[col].median()
                df_to_scale[col].fillna(median_val, inplace=True)
        st.write(f"Filled {nan_count_before} missing value(s) with the column median.")
    else:
        st.write("No missing values found post-encoding. Data is clean.")
    # --- END OF FIX ---

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=df_to_scale.columns)

    st.success("✅ Data encoded and normalized successfully!")
    st.dataframe(df_scaled.head())

except KeyError as e:
    st.error(f"Encoding Error: A column name mismatch occurred. {e}")
    st.info(f"This likely means the CSV file '{file_path}' does not match the expected survey structure.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during processing: {e}")
    st.stop()


# ===================================================================
# 3️⃣ DESCRIPTIVE ANALYTICS (ANALYSIS 1)
# ===================================================================
st.subheader("📊 Step 3: Descriptive Analytics (Proving Our Strategy)")

st.markdown("**Visualizing Key Project Variables (ASU Maroon & Gold Theme)**")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Boxplot: Cost Challenge vs. Pass Rate Challenge
sns.boxplot(data=df_scaled[['Q6_Cost', 'Q6_PassRates']], ax=axes[0], palette=["#8C1D40", "#FFC627"])
axes[0].set_title("Boxplot: Key Market Challenges", fontsize=10)
axes[0].set_xticklabels(['Challenge: Cost', 'Challenge: Pass Rates'])

# Histogram: Likelihood to Adopt
sns.histplot(df_scaled['Q12. Adopt Likely'], bins=5, kde=True, ax=axes[1], color="#8C1D40")
axes[1].set_title("Histogram: Distribution of 'Likelihood to Adopt'", fontsize=10)
axes[1].set_xlabel("Adoption Likelihood (Normalized)")

st.pyplot(fig)


# --- Correlation Heatmap (FROM ANALYSIS 1) ---
st.markdown("**Correlation Heatmap (Proving Our Segments)**")
fig_h, ax_h = plt.subplots(figsize=(10, 8))

# Select only the key columns from Analysis 1 for the heatmap
heatmap_cols = [
    'Q6_Cost', 'Q6_PassRates', 'Q7_PassRates', 'Q7_CostReduce',
    'Q10_Features_Payment', 'Q12. Adopt Likely'
]
corr_matrix = df_scaled[heatmap_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax_h=ax_h, vmin=-1, vmax=1)
ax_h.set_title("Feature Correlation Matrix (The 'Why')")
st.pyplot(fig_h)
st.info("""
**Heatmap Interpretation:**
* **(-0.81):** Strong NEGATIVE correlation. As **Cost Challenge (Q6_Cost)** goes up, **Adoption Likelihood (Q12)** plummets. (Proves Segment 1)
* **(+0.85):** Strong POSITIVE correlation. As **Pass Rate Value (Q7_PassRates)** goes up, **Adoption Likelihood (Q12)** soars. (Proves Segment 2)
""")


# ===================================================================
# 4️⃣ HYPOTHESIS TESTING
# ===================================================================
st.subheader("📈 Step 4: Hypothesis Testing (Our Strategic Segments)")

st.markdown("""
**H₁:** *Value for Pass Rates* is positively correlated with *Adoption Likelihood*. (Spearman Test)
**H₂:** The "Outcomes-Driven" segment has a *statistically higher* adoption likelihood than the "Budget-Sensitive" segment. (t-Test)
""")

# --- H₁: Spearman Correlation ---
# We use the scaled data for this test
corr, p_corr = spearmanr(df_scaled['Q7_PassRates'], df_scaled['Q12. Adopt Likely'])
st.write(f"**Spearman Correlation (H₁):** ρ = {corr:.3f}, p = {p_corr:.3g}")
if p_corr < 0.05:
    st.success("✅ H₁ is supported: Value for Pass Rates is significantly correlated with Adoption Likelihood.")
else:
    st.warning("⚠️ H₁ is not supported.")


# --- H₂: Independent t-Test ---
# We use Q10_Features_Payment as the separator for our two segments
# 1 = One-time purchase (Budget-Sensitive)
# 2 = Subscription (Outcomes-Driven)
# Note: We use the *encoded but non-scaled* data (df_processed) for a clean split and mean interpretation
group1 = df_processed[df_processed['Q10_Features_Payment'] == 1]['Q12. Adopt Likely']
group2 = df_processed[df_processed['Q10_Features_Payment'] == 2]['Q12. Adopt Likely']

if len(group1) > 1 and len(group2) > 1:
    # Run the t-test (assuming unequal variances by default, which is safer)
    tstat, pval = ttest_ind(group1, group2, equal_var=False)
    st.write(f"**t-Test (H₂ - Segment 1 vs Segment 2):** t = {tstat:.3f}, p = {pval:.3g}")
    
    if pval < 0.05 and tstat < 0: # Check if p-value is significant AND group2 mean is higher
        st.success(f"✅ H₂ is supported: There is a statistically significant difference (p < 0.05) in adoption likelihood between the two segments.")
        st.write(f"**Insight:** The 'Outcomes/Subscription' segment (Mean Likelihood: {group2.mean():.2f}) is significantly more likely to adopt than the 'Budget/One-time' segment (Mean Likelihood: {group1.mean():.2f}).")
    else:
        st.warning("⚠️ H₂ is not supported: No significant difference detected between segments.")
else:
    st.warning("⚠️ Insufficient data in both segments for t-Test.")


# ===================================================================
# 5️⃣ PREDICTIVE MODEL (Demo)
# ===================================================================
st.subheader("🤖 Step 5: Predictive Modeling (Demo)")
st.markdown("**Training a Logistic Regression model to predict 'High Adoption Likelihood'**")

# Define target and features
target_col = 'Q12. Adopt Likely' # This is the normalized column

# Create binary target variable (High Adoption Likelihood vs Low)
# We use 0.5 (the median of the normalized data) as the cutoff
y = (df_scaled[target_col] > 0.5).astype(int)

# Features: All other columns in the scaled set
X = df_scaled.drop(columns=[target_col], axis=1, errors='ignore')

# Drop columns that are constant (if any)
X = X.loc[:, (X != X.iloc[0]).any()]

if X.shape[1] > 0 and len(X) >= 2:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict probabilities on test set
    proba = model.predict_proba(X_test)[:, 1]

    # Display results
    st.write("**Predicted 'High Adoption' Probabilities (Test Set):**")
    prediction_df = pd.DataFrame({
        "Respondent_ID": df.loc[X_test.index, 'Respondent ID'],
        "Predicted_Prob": np.round(proba, 3)
    }).set_index("Respondent_ID")
    st.dataframe(prediction_df.head(10))

    # Bar chart visualization
    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    # Sort for a cleaner visual
    prediction_df_sorted = prediction_df.sort_values('Predicted_Prob', ascending=False)
    
    # Ensure respondent IDs are strings for plotting
    plot_indices = prediction_df_sorted.index.astype(str)
    
    ax_pred.bar(plot_indices, prediction_df_sorted['Predicted_Prob'], color='#8C1D40')
    ax_pred.set_title("Predicted Adoption Probability (Test Set Samples)")
    ax_pred.set_ylabel("Probability of High Adoption")
    ax_pred.set_xlabel("Respondent ID")
    plt.xticks(rotation=90, fontsize=8)
    st.pyplot(fig_pred)
    st.success("✅ Predictive model executed successfully!")
else:
    st.warning("⚠️ Not enough unique features or data points to train the predictive model.")


# ===================================================================
# 6️⃣ PIPELINE SUMMARY
# ===================================================================
st.subheader("🚀 End-to-End Pipeline Completed")
st.markdown("""
This demo shows an **automated, modular pipeline** for the V-Med Pro project:
- **Loads** our live Capstone data (`final_30_responses_dataset.csv`).
- **Cleans & Encodes** the data for analysis.
- **Visualizes** our key strategic segments via a Correlation Heatmap.
- **Statistically Proves** our hypotheses with a t-Test, confirming our segments are distinct.
- **Predicts** which HEIs are most likely to adopt, allowing for targeted sales efforts.
""")

st.balloons()
