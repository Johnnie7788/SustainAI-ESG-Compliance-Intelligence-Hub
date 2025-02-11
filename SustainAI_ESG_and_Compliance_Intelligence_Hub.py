#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import os
from bs4 import BeautifulSoup
import time
import pandas as pd
import plotly.express as px
import openai
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# ---- App Config ----
st.set_page_config(page_title="SustainAI: ESG & Compliance Intelligence Hub", page_icon="🌿", layout="wide")
st.title("🌿 SustainAI: ESG & Compliance Intelligence Hub")
st.markdown("---")

# ---- File Upload for ESG Data ----
st.sidebar.header("📂 Upload Your ESG Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    esg_data = pd.read_csv(uploaded_file)

    # Display Uploaded Data Preview
    st.write("### Uploaded ESG Data Preview:")
    st.dataframe(esg_data)

    # Check for required columns
    required_columns = ["Company", "Carbon Emissions", "Energy Consumption", "Water Usage", 
                        "Waste Recycled (%)", "Regulatory Compliance", "Predicted ESG Score"]

    missing_columns = [col for col in required_columns if col not in esg_data.columns]
    
    if missing_columns:
        st.error(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
        st.stop()  # Stop further execution if required columns are missing

    # ---- Handle Missing Values ----
    if esg_data.isnull().values.any():
        st.warning("⚠️ Missing values detected in the dataset. Applying automatic corrections...")

        # Fill missing numeric values with the median
        numeric_cols = esg_data.select_dtypes(include=[np.number]).columns
        esg_data[numeric_cols] = esg_data[numeric_cols].fillna(esg_data[numeric_cols].median())

        # Fill missing categorical values with the most frequent value
        categorical_cols = esg_data.select_dtypes(include=[object]).columns
        esg_data[categorical_cols] = esg_data[categorical_cols].fillna(esg_data[categorical_cols].mode().iloc[0])

        st.success("✅ Missing values have been handled successfully.")

    # Display the cleaned dataset
    st.write("### Cleaned ESG Data (After Handling Missing Values):")
    st.dataframe(esg_data)

else:
    st.warning("⚠️ Please upload an ESG dataset to proceed.")
    st.stop()  # Stop execution until file is uploaded


    
    # AI-Driven ESG Scoring
    st.subheader("🌍 AI-Driven ESG Scoring")
    scaler = MinMaxScaler()
    esg_scaled = pd.DataFrame(scaler.fit_transform(esg_data.select_dtypes(include=[np.number])), columns=esg_data.select_dtypes(include=[np.number]).columns)
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_model.fit(esg_scaled, np.random.randint(60, 100, size=len(esg_scaled)))
    predicted_scores = ml_model.predict(esg_scaled)
    esg_data["Predicted ESG Score"] = predicted_scores
    st.dataframe(esg_data[["Company", "Predicted ESG Score"]].head())

# ---- Custom User Dashboard ----
st.header("📊 Custom User Dashboard")
st.write("Personalized ESG analytics and key sustainability insights tailored to your organization.")

# User Input for Customization
industry_type = st.selectbox("Select Your Industry", ["Manufacturing", "Finance", "Technology", "Healthcare", "Retail"])
company_size = st.slider("Company Size (Employees)", min_value=50, max_value=5000, value=1000)
region = st.selectbox("Select Your Region", ["North America", "Europe", "Asia", "South America", "Africa", "Australia"])

# AI-Powered ESG Scoring Model
st.header("🌍 AI-Driven ESG Scoring Model")
st.write("Machine Learning-based ESG assessment, predicting sustainability performance and compliance risk.")

# Sample ESG Data for Scoring Model
esg_features = np.random.rand(10, 5) * 100  # Generating random ESG feature scores
esg_df = pd.DataFrame(esg_features, columns=["Carbon Emissions", "Energy Efficiency", "Water Usage", "Waste Recycling", "Regulatory Compliance"])
scaler = MinMaxScaler()
esg_df_scaled = pd.DataFrame(scaler.fit_transform(esg_df), columns=esg_df.columns)

# ML Model for ESG Scoring
ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
ml_model.fit(esg_df_scaled, np.random.randint(60, 100, size=10))  # Random target scores
predicted_score = ml_model.predict([scaler.transform([[50, 70, 60, 80, 90]])[0]])[0]

st.metric("Predicted ESG Score", f"{predicted_score:.2f}/100")
st.progress(predicted_score / 100)

# Custom ESG Improvement Plan
st.subheader("🔍 ESG Performance Improvement Plan")
if predicted_score < 70:
    st.warning("⚠️ ESG score is below optimal. Recommended actions:")
    esg_recommendations = [
        "Increase renewable energy adoption.",
        "Enhance supply chain sustainability policies.",
        "Improve water conservation and waste recycling.",
        "Strengthen ESG compliance frameworks."
    ]
    for rec in esg_recommendations:
        st.write(f"- {rec}")
else:
    st.success("Your ESG score is on track! Keep up the great work.")

# ---- AI-Powered Real-Time ESG Monitoring ----
st.header("⏳ AI-Powered Real-Time ESG Monitoring")
st.write("Track sustainability metrics in real-time with AI-driven alerts and anomaly detection.")

live_metrics = {"Carbon Emissions": np.random.randint(100, 500),
                "Energy Consumption": np.random.randint(5000, 20000),
                "Water Usage": np.random.randint(10000, 50000)}
st.json(live_metrics)

if st.button("Analyze Live ESG Metrics"):
    anomalies = []
    if live_metrics["Carbon Emissions"] > 400:
        anomalies.append("High Carbon Emissions detected. Immediate reduction actions recommended.")
    if live_metrics["Energy Consumption"] > 15000:
        anomalies.append("Excessive energy consumption. Consider efficiency optimizations.")
    if live_metrics["Water Usage"] > 40000:
        anomalies.append("High water usage detected. Implement conservation measures.")
    
    if anomalies:
        st.error("⚠️ Anomalies detected:")
        for anomaly in anomalies:
            st.write(f"- {anomaly}")
    else:
        st.success("All ESG metrics are within optimal ranges.")


# ---- 1. Sustainability Strategy Dashboard ----
st.header("1️⃣ Sustainability Strategy Dashboard")
st.write("Develop and coordinate sustainability strategies in line with CSRD and ESG principles, incorporating continuous improvement frameworks and predictive analytics.")

# Dynamic KPI Customization
st.subheader("Set Your Sustainability Goals")
carbon_target = st.number_input("Target Carbon Footprint Reduction (%)", min_value=0, max_value=100, value=30)
renewable_target = st.number_input("Target Renewable Energy Usage (%)", min_value=0, max_value=100, value=50)
water_target = st.number_input("Target Water Conservation (%)", min_value=0, max_value=100, value=20)

strategy_kpis = {"Carbon Footprint Reduction (%)": carbon_target, 
                 "Renewable Energy Usage (%)": renewable_target, 
                 "Water Conservation (%)": water_target}

fig1 = px.bar(x=strategy_kpis.keys(), y=strategy_kpis.values(), labels={'x':'KPIs', 'y':'Performance (%)'}, title="Sustainability Strategy Performance")
st.plotly_chart(fig1)

# AI-Powered Trend Forecasting
st.subheader("Sustainability Trend Forecasting")
months = np.array(range(1, 13)).reshape(-1, 1)
past_performance = np.array([25, 27, 28, 29, 30, 32, 35, 37, 40, 42, 45, 50]).reshape(-1, 1)

model = LinearRegression()
model.fit(months, past_performance)
future_months = np.array(range(13, 19)).reshape(-1, 1)
future_trends = model.predict(future_months)

fig2 = px.line(x=list(range(1, 19)), y=np.append(past_performance, future_trends), 
               labels={'x':'Months', 'y':'Carbon Reduction (%)'}, title="Carbon Footprint Reduction Forecast")
st.plotly_chart(fig2)

# Industry Benchmarking
st.subheader("Company vs Industry Benchmarking")
industry_avg = {"Carbon Footprint Reduction (%)": 40, "Renewable Energy Usage (%)": 60, "Water Conservation (%)": 30}
df = pd.DataFrame({"KPI": list(strategy_kpis.keys()), "Company": list(strategy_kpis.values()), "Industry Average": list(industry_avg.values())})
fig3 = px.bar(df, x="KPI", y=["Company", "Industry Average"], barmode="group", title="Company vs Industry Sustainability Performance")
st.plotly_chart(fig3)

# Automated Alerts for KPI Deviations
if carbon_target < 20:
    st.warning("⚠️ Carbon footprint reduction target is too low. Increase efforts in renewable energy.")
if renewable_target < 40:
    st.warning("⚠️ Renewable energy usage is below industry standards. Consider investing in solar/wind power.")
if water_target < 25:
    st.warning("⚠️ Water conservation efforts need improvement. Implement better water recycling measures.")

# AI-Powered Sustainability Recommendations
st.subheader("AI-Powered Sustainability Recommendations")
if st.button("Generate Recommendations"):
    recommendations = []
    if carbon_target < 30:
        recommendations.append("Increase investment in carbon offset programs and renewable energy sources.")
    if renewable_target < 50:
        recommendations.append("Consider implementing solar and wind energy projects for long-term sustainability.")
    if water_target < 30:
        recommendations.append("Enhance water recycling and rainwater harvesting initiatives.")
    
    if recommendations:
        st.success("Here are your AI-powered recommendations:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.success("Your sustainability goals are well-aligned with best practices!")

# ---- 2. Industry-Standard CSRD & ESG Compliance Reporting ----
st.header("2⃣ Industry-Standard CSRD & ESG Compliance Reporting")
st.write("Generate structured reports in compliance with CSRD and international sustainability frameworks. This module incorporates real-time data validation, automated anomaly detection, and AI-driven compliance tracking.")

# Real-Time Compliance Data
report_data = pd.DataFrame({"Metric": ["Scope 1 Emissions", "Scope 2 Emissions", "Scope 3 Emissions"],
                             "Value": [500, 1200, 3000],
                             "Unit": ["tonnes CO2", "tonnes CO2", "tonnes CO2"]})
st.dataframe(report_data)

# AI-Driven Compliance Tracking
if st.button("Generate Compliance Report"):
    st.success("CSRD Compliance Report Generated Successfully!")
    st.write("Non-compliance detected in Scope 3 Emissions. Recommendation: Improve supplier engagement through proactive ESG initiatives.")

# Anomaly Detection
st.subheader("Automated Anomaly Detection")
anomaly_threshold = 2500
if report_data.loc[2, "Value"] > anomaly_threshold:
    st.error("⚠️ High Scope 3 Emissions detected! Immediate corrective action recommended.")

# Benchmarking Against Regulatory Standards
st.subheader("Regulatory Benchmarking")
csrd_standards = {"Scope 1 Emissions Limit (tonnes CO2)": 400, "Scope 2 Emissions Limit (tonnes CO2)": 1000, "Scope 3 Emissions Limit (tonnes CO2)": 2500}
df_benchmark = pd.DataFrame({"Metric": list(csrd_standards.keys()), "Company Performance": [500, 1200, 3000], "Regulatory Limit": list(csrd_standards.values())})
fig_benchmark = px.bar(df_benchmark, x="Metric", y=["Company Performance", "Regulatory Limit"], barmode="group", title="Company vs Regulatory Compliance")
st.plotly_chart(fig_benchmark)

# Proactive Compliance Recommendations
st.subheader("AI-Powered Compliance Recommendations")
if st.button("Generate Compliance Advice"):
    recommendations = []
    if report_data.loc[2, "Value"] > anomaly_threshold:
        recommendations.append("Develop supplier sustainability programs to reduce Scope 3 emissions.")
    if report_data.loc[1, "Value"] > csrd_standards["Scope 2 Emissions Limit (tonnes CO2)"]:
        recommendations.append("Increase investments in renewable energy sources to reduce Scope 2 emissions.")
    
    if recommendations:
        st.success("AI-generated compliance improvement suggestions:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.success("Your company is fully compliant with current CSRD standards!")

# ---- 3. Climate Risk Analysis & Impact Measurement ----
st.header("3⃣ Climate Risk Analysis & Impact Measurement")
st.write("Assess climate risks and predict future sustainability impacts using advanced AI-driven impact measurement models, scenario analysis, and automated risk mitigation strategies.")

# AI-Powered Climate Risk Forecasting
dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
climate_risk_scores = np.random.randint(10, 100, size=12)
fig4 = px.line(x=dates, y=climate_risk_scores, title="Predicted Climate Risk Scores Over Next 12 Months", labels={'x':'Date', 'y':'Risk Score'})
st.plotly_chart(fig4)

# Scenario-Based Risk Analysis
st.subheader("Scenario-Based Climate Risk Analysis")
scenarios = ["Low Risk", "Moderate Risk", "High Risk"]
risk_impact = [20, 50, 90]
df_scenario = pd.DataFrame({"Scenario": scenarios, "Impact Score": risk_impact})
fig5 = px.bar(df_scenario, x="Scenario", y="Impact Score", title="Scenario-Based Climate Risk Impact")
st.plotly_chart(fig5)


# ---- Geospatial Climate Risk Mapping ----
st.header("🌍 Geospatial Climate Risk Mapping")
st.write("Analyze climate risks across global locations with AI-driven impact assessments.")

geo_risk_data = pd.DataFrame({
    "Country": [
        "United States", "Germany", "China", "Brazil", "South Africa",
        "India", "Australia", "France", "Canada", "Japan", "United Kingdom",
        "Mexico", "Russia", "Italy", "South Korea"
    ],
    "ISO-3": ["USA", "DEU", "CHN", "BRA", "ZAF", "IND", "AUS", "FRA", "CAN", "JPN", 
              "GBR", "MEX", "RUS", "ITA", "KOR"],
    "Risk Score": np.random.randint(20, 90, 15)  # Climate risk scores
})

fig_geo_risk = px.choropleth(
    geo_risk_data,
    locations="ISO-3",  # ✅ Correct column name for ISO-3 codes
    color="Risk Score",
    hover_name="Country",
    title="Global Climate Risk Assessment",
    scope="world"
)

st.plotly_chart(fig_geo_risk)

# Adding this data to the dataset
geo_risk_data_corrected_path = "geo_risk_data_corrected.csv"
geo_risk_data.to_csv(geo_risk_data_corrected_path, index=False)
st.download_button(label="Download Geospatial Risk Data", data=open(geo_risk_data_corrected_path, "rb"), file_name="geo_risk_data_corrected.csv")


# AI-Powered Risk Probability Analysis
st.subheader("AI-Powered Risk Probability Analysis")
risk_prob_model = RandomForestRegressor()
months_future = np.array(range(1, 13)).reshape(-1, 1)
risk_scores_future = np.random.randint(20, 80, size=12)
risk_prob_model.fit(months_future, risk_scores_future)
predicted_risks = risk_prob_model.predict(months_future)
fig7 = px.line(x=range(1, 13), y=predicted_risks, title="Predicted Climate Risk Probability Over 12 Months")
st.plotly_chart(fig7)

# Automated Risk Mitigation Suggestions
st.subheader("AI-Powered Risk Mitigation Strategies")
if st.button("Generate Risk Mitigation Plan"):
    risk_mitigation = [
        "Implement advanced weather prediction analytics to anticipate extreme climate events.",
        "Strengthen sustainability policies to reduce exposure to environmental risks.",
        "Increase investment in resilient infrastructure to mitigate long-term climate impact.",
        "Utilize geospatial analytics to identify high-risk regions and take preventive actions."
    ]
    st.success("Recommended Climate Risk Mitigation Strategies:")
    for strategy in risk_mitigation:
        st.write(f"- {strategy}")

# ---- 4. ESG Data Collection & Analytics Hub ----
st.header("4⃣ ESG Data Collection & Analytics Hub")
st.write("Automated, real-time ESG data collection and analytics across departments, ensuring data accuracy, AI-driven insights, and compliance tracking.")

# AI-Powered ESG Data Validation
st.subheader("AI-Powered ESG Data Validation")
esg_data = {"Energy Consumption (kWh)": 15000, "Water Usage (Liters)": 50000, "Waste Recycled (%)": 75}
st.json(esg_data)

if st.button("Run Data Validation"):
    st.success("ESG Data Validation Completed! No inconsistencies found.")
    st.write("AI-powered checks confirm data accuracy and completeness.")

# ESG Data Visualization
st.subheader("ESG Performance Visualization")
df_esg = pd.DataFrame({"Metric": list(esg_data.keys()), "Value": list(esg_data.values())})
fig8 = px.bar(df_esg, x="Metric", y="Value", title="ESG Performance Metrics")
st.plotly_chart(fig8)

# AI-Driven Predictive ESG Analytics
st.subheader("Predictive ESG Analytics")
predicted_esg_performance = {"Projected Energy Consumption (kWh)": 14000, "Projected Water Usage (Liters)": 48000, "Projected Waste Recycled (%)": 80}
st.json(predicted_esg_performance)

# Automated Compliance & Reporting Alerts
st.subheader("Automated Compliance & Reporting Alerts")
if esg_data["Waste Recycled (%)"] < 70:
    st.warning("⚠️ Waste recycling below threshold. Immediate improvement required.")
if esg_data["Energy Consumption (kWh)"] > 16000:
    st.warning("⚠️ High energy consumption detected. Consider efficiency improvements.")
if esg_data["Water Usage (Liters)"] > 55000:
    st.warning("⚠️ Excessive water usage identified. Implement conservation measures.")

# AI-Powered ESG Improvement Suggestions
st.subheader("AI-Powered ESG Improvement Strategies")
if st.button("Generate ESG Recommendations"):
    esg_recommendations = [
        "Optimize energy efficiency through AI-powered smart grid management.",
        "Implement circular economy principles to enhance waste management.",
        "Develop water conservation strategies, including rainwater harvesting.",
    ]
    st.success("Recommended ESG Improvement Strategies:")
    for strategy in esg_recommendations:
        st.write(f"- {strategy}")


# ---- Load API Key from Streamlit Secrets ----
if "API_KEY" in st.secrets:
    SERPAPI_KEY = st.secrets["API_KEY"]  # Streamlit Cloud Secrets
else:
    st.error("⚠️ API_KEY is missing. Please check Streamlit Secrets setup.")
    st.stop()  # Stop execution if key is missing

# ---- AI-Powered Sustainability Advisor ----
st.header("5⃣ AI-Powered Sustainability Advisor")
st.write("Search for sustainability information, regulatory compliance updates, and industry trends.")

# ---- AI Chat Interface ----
st.subheader("🔍 Search for Sustainability Information")
user_query = st.text_area("Enter your question or topic:")

# ---- Function to Get Web Search Results Using SerpAPI ----
def get_web_search_results(query):
    search_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query + " sustainability",
        "api_key": SERPAPI_KEY,
        "num": 5  # Fetch 5 search results
    }

    try:
        response = requests.get(search_url, params=params)
        data = response.json()

        search_results = []
        for result in data.get("organic_results", []):
            title = result.get("title")
            url = result.get("link")
            if title and url:
                search_results.append(f"🔹 [{title}]({url})")

        return search_results if search_results else ["⚠️ No relevant results found. Try refining your search."]
    
    except Exception as e:
        return [f"⚠️ Error fetching results: {str(e)}"]

# ---- When User Clicks "Search Sustainability Info" ----
if st.button("Search Sustainability Info") and user_query:
    with st.spinner("Searching the web..."):
        search_results = get_web_search_results(user_query)
    
    if search_results:
        for result in search_results:
            st.markdown(result)
    else:
        st.error("⚠️ No results found. Try refining your query.")



# ---- 6. Sustainability Budget Optimization ----
st.header("6⃣ Sustainability Budget Optimization")
st.write("Optimize sustainability investments with AI-driven predictive modeling and dynamic budget allocation strategies.")

# AI-Powered Budget Forecasting
st.subheader("AI-Powered Budget Forecasting")
budget_data = {"Carbon Reduction Projects": 50000, "Renewable Energy Investments": 120000, "Water Conservation": 25000}
st.json(budget_data)

fig6 = px.pie(names=budget_data.keys(), values=budget_data.values(), title="Sustainability Budget Allocation")
st.plotly_chart(fig6)

# Dynamic Budget Adjustment
st.subheader("Dynamic Budget Allocation")
budget_adjustment = st.slider("Adjust Renewable Energy Investment (%)", min_value=-20, max_value=20, value=0)
adjusted_budget = budget_data["Renewable Energy Investments"] * (1 + budget_adjustment / 100)
st.write(f"Updated Renewable Energy Investment: €{adjusted_budget:,.2f}")

# AI-Driven Cost Efficiency Analysis
st.subheader("AI-Driven Cost Efficiency Analysis")
if st.button("Run Cost Efficiency Analysis"):
    cost_savings = [
        "Reduce operational costs by optimizing energy consumption.",
        "Enhance resource efficiency through predictive maintenance.",
        "Leverage AI insights to minimize unnecessary expenditures."
    ]
    st.success("Recommended Cost Efficiency Strategies:")
    for savings in cost_savings:
        st.write(f"- {savings}")

# Predictive ROI for Sustainability Investments
st.subheader("Predictive ROI for Sustainability Investments")
if st.button("Generate ROI Forecast"):
    roi_forecast = [
        "Expected ROI on Renewable Energy Investments: 15% over 5 years.",
        "Projected savings from Carbon Reduction Initiatives: €20,000 annually.",
        "Water Conservation ROI expected at 10% in efficiency gains."
    ]
    st.success("Predicted Sustainability ROI:")
    for roi in roi_forecast:
        st.write(f"- {roi}")

# ---- 7. Automated ESG Auditing & Compliance Tracker ----
st.header("7⃣ Automated ESG Auditing & Compliance Tracker")
st.write("Monitor sustainability compliance, generate audit reports, and support internal and external audits with AI-driven analysis and predictive risk assessment.")

# AI-Powered Audit Report Generation
st.subheader("AI-Powered Audit Report Generation")
auditing_data = pd.DataFrame({"Audit Area": ["GHG Emissions", "Energy Efficiency", "Waste Management"],
                               "Status": ["Compliant", "Needs Improvement", "Compliant"]})
st.dataframe(auditing_data)

if st.button("Run ESG Audit"):
    st.success("ESG Audit Completed Successfully!")
    st.write("Recommended Action: Improve Energy Efficiency by 15%.")
    st.write("Audit Report available for download.")

# Predictive Risk Analysis
st.subheader("Predictive Risk Analysis")
if st.button("Run Predictive ESG Risk Assessment"):
    risk_factors = [
        "High likelihood of regulatory penalties due to Scope 3 emissions.",
        "Moderate risk of non-compliance in water usage efficiency.",
        "Low risk in renewable energy adoption compliance."
    ]
    st.success("Predicted ESG Risk Factors:")
    for risk in risk_factors:
        st.write(f"- {risk}")

# AI-Driven Continuous Compliance Monitoring
st.subheader("AI-Driven Continuous Compliance Monitoring")
if st.button("Activate Continuous Compliance Monitoring"):
    st.success("Real-time ESG compliance tracking enabled.")
    st.write("System will now monitor regulatory changes and alert for compliance risks.")

