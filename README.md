# 🥽 V-Med Pro: VR Value Proposition and Product Positioning 🚑

**Company:** [Vizitech USA](https://www.vizitechusa.com/)  
**Product:** V-Med Pro (Virtual Reality Medical Trainer)  

This repository hosts the analytics suite developed for **Vizitech USA**, a leader in 3D and VR education technology. The project analyzes performance data and user feedback from the **V-Med Pro** system, a VR training solution designed for Emergency Medical Technicians (EMTs) and Paramedics.

## 📦 Repository Structure

### 📂 app/
* `analytics_app.py` - The main Streamlit dashboard application code.
* `VMedPro_Demo.ipynb` - Prototype notebook for the application logic.

### 📂 analysis/
* `1_Statistical_Correlation.ipynb` - Analysis linking VR usage patterns to learner performance.
* `2_Sentiment_Analysis_Part1.ipynb` - Initial NLP modeling of user feedback.
* `3_Sentiment_Analysis_Part2.ipynb` - Advanced sentiment classification and insights.

### 📂 docs/
* `Report.pdf` - Complete documentation of methodology and results.
* `Presentation_Deck.pdf` - Executive summary and slide deck.
* `Project_Poster.pdf` - Visual summary of the project for exhibitions.
* `Demo_Walkthrough.pdf` - Step-by-step guide to the V-Med Pro dashboard.
* `Power BI Dashboard.pdf` - Static export of the Business Intelligence report.

### 📂 Dataset
* `processed_dataset.csv` - Anonymized real time dataset used for modeling and dashboard generation.

## 🔍 Project Scope & Solution

### 1. The Business Challenge
Vizitech's V-Med Pro offers immersive medical scenarios (e.g., trauma response, anatomy). The company needed a unified analytics layer to:
* **Validate Efficacy:** Statistically prove that VR training improves learner outcomes.
* **Monitor Sentiment:** aggregating qualitative feedback from users to guide future VR scenario development.

### 2. The Solution: V-Med Pro Analytics Hub
We developed a multi-module analytics engine:
* **Statistical Engine:** Performed Pearson/Spearman correlation analysis to link specific VR usage patterns (e.g., "Time in Scenario") with performance metrics.
* **NLP Engine:** Scraped and analyzed text feedback using NLTK to categorize user sentiment (Positive/Negative/Neutral) regarding hardware comfort and software usability.
* **Visualization Hub:** A centralized dashboard integrating **Streamlit** (Python) and **Power BI** to present these insights to non-technical executives.

## 📊 Key Insights
* **Usage vs. Mastery:** Found strong positive correlations between repeated VR scenario attempts and final assessment scores.
* **User Feedback:** Sentiment analysis identified specific VR modules that users found most engaging versus those requiring UI improvements.

## 🛠️ Technology Stack
* **Language:** Python 3.9+
* **Dashboarding:** Streamlit, Power BI.
* **Data Science:** Pandas, NumPy, SciPy (Stats), Matplotlib.
* **NLP:** NLTK, TextBlob.

## 🚀 How to Run the Dashboard
1.  Clone the repository
   
2.  Install dependencies
    ```bash
    pip install streamlit pandas numpy seaborn matplotlib nltk textblob
    ```
    
3.  Navigate to the `app/` folder and run
    ```bash
    streamlit run app.py
    ```

---
## 🏆 Team & Recognition
This project was successfully delivered to **Vizitech Solutions** as part of the MS Business Analytics Capstone.

* **Corporate Feature:** Our team was recognized for this contribution on the Vimana Consulting Corporate Responsibility page.
* 🔗 **View Team Feature:** [Click here to see our team photo and project mention](http://www.vimanacon.com/corporate-responsibility-and-development.html)
-----
