import streamlit as st
import pandas as pd
import requests
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(
    page_title="Exit Interview API ",
   
    layout="wide",
    initial_sidebar_state="expanded"
)


API_URL = "http://localhost:5000/api"
# Sidebar header


# Function to make API requests
def make_api_request(endpoint, method="GET", data=None):
    """Make a request to the API and handle errors"""
    url = f"{API_URL}/{endpoint}"
    
    try:
        with st.spinner(f"Making {method} request to {endpoint}..."):
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, headers=headers, data=json.dumps(data))
            else:
                return {"status": "error", "message": f"Unsupported method: {method}"}
                
            return response.json()
    except Exception as e:
        return {"status": "error", "message": f"Error connecting to API: {str(e)}"}

# Function to check API health
def check_api_health():
    return make_api_request("health")

# Function to get metadata
def get_metadata():
    return make_api_request("metadata")

# Main app
def main():
    # Header
    st.title("Exit Interview Analysis")
    
    # Check API connection
    health_check = check_api_health()
    
    if health_check.get("status") == "success":
        st.success(f"✅ API Connection Successful: {health_check.get('message', '')}")
        
        # Get metadata
        metadata = get_metadata()
        if metadata.get("status") == "success":
            # Create tabs
            tab1, tab2, = st.tabs(["Analyze Data", "Generate Reports",])
            
            # Tab 1: Analyze Data
            with tab1:
                st.header("Analyze Exit Interview Data")
                
                # Filter selection
                st.subheader("Filter Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    department = st.selectbox(
                        "Department",
                        options=["All"] + metadata.get("departments", [])
                    )
                
                with col2:
                    role = st.selectbox(
                        "Role",
                        options=["All"] + metadata.get("roles", [])
                    )
                
                with col3:
                    tenure = st.selectbox(
                        "Tenure",
                        options=["All"] + metadata.get("tenure_categories", [])
                    )
                
                # Analysis button
                if st.button("Analyze Data", type="primary"):
                    # Prepare filter data
                    filter_data = {}
                    if department != "All":
                        filter_data["department"] = department
                    if role != "All":
                        filter_data["role"] = role
                    if tenure != "All":
                        filter_data["tenure"] = tenure
                    
                    # Make API request
                    analysis_result = make_api_request("analyze", method="POST", data=filter_data)
                    
                    if analysis_result.get("status") == "success":
                        insights = analysis_result.get("insights", {})
                        
                        # Display statistics
                        st.subheader("Analysis Results")
                        st.markdown(f"**Filter Applied:** {insights.get('filter_description', 'All Data')}")
                        st.markdown(f"**Records Analyzed:** {insights.get('record_count', 0)}")
                        
                        # Display heatmap
                        if "heatmap" in insights:
                            st.subheader("Exit Reasons")
                            heatmap_data = insights["heatmap"]
                            st.image(f"data:image/png;base64,{heatmap_data}")
                        
                        # Display key themes
                        if "key_themes" in insights:
                            st.subheader("Key Themes")
                            themes_container = st.container()
                            
                            for i, theme in enumerate(insights["key_themes"], 1):
                                themes_container.markdown(f"**Theme {i}:** {', '.join(theme)}")
                        
                        # Display category percentages
                        if "category_percentages" in insights:
                            st.subheader("Exit Reasons Breakdown")
                            
                            # Convert to DataFrame for better display
                            percentages = insights["category_percentages"]
                            df_percentages = pd.DataFrame({
                                "Reason": [" ".join(r.split("_")).title() for r in percentages.keys()],
                                "Percentage": list(percentages.values())
                            }).sort_values(by="Percentage", ascending=False)
                            
                            # Only show non-zero values
                            df_percentages = df_percentages[df_percentages["Percentage"] > 0]
                            
                            # Display as table
                            st.dataframe(
                                df_percentages,
                                column_config={
                                    "Percentage": st.column_config.ProgressColumn(
                                        "Percentage (%)",
                                        format="%.1f%%",
                                        min_value=0,
                                        max_value=100,
                                    )
                                },
                                hide_index=True
                            )
                        
                        # Display AI insights
                        if "llm_report" in insights:
                            st.subheader("AI-Enhanced Insights")
                            st.markdown(insights["llm_report"])
                    else:
                        st.error(f"Error: {analysis_result.get('message', 'Unknown error')}")
            
            # Tab 2: Generate Reports
            with tab2:
                st.header("Generate Exit Interview Reports")
                
                # Filter selection
                st.subheader("Filter Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    report_department = st.selectbox(
                        "Department",
                        options=["All"] + metadata.get("departments", []),
                        key="report_dept"
                    )
                
                with col2:
                    report_role = st.selectbox(
                        "Role",
                        options=["All"] + metadata.get("roles", []),
                        key="report_role"
                    )
                
                with col3:
                    report_tenure = st.selectbox(
                        "Tenure",
                        options=["All"] + metadata.get("tenure_categories", []),
                        key="report_tenure"
                    )
                
                # Generate report button
                if st.button("Generate Comprehensive Report", type="primary"):
                    # Prepare filter data
                    filter_data = {}
                    if report_department != "All":
                        filter_data["department"] = report_department
                    if report_role != "All":
                        filter_data["role"] = report_role
                    if report_tenure != "All":
                        filter_data["tenure"] = report_tenure
                    
                    # Make API request with progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Requesting report generation...")
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    report_result = make_api_request("report", method="POST", data=filter_data)
                    
                    progress_bar.progress(75)
                    status_text.text("Processing report...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(100)
                    status_text.text("Report ready!")
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if report_result.get("status") == "success":
                        report_text = report_result.get("report", "")
                        
                        # Display report
                        st.subheader("Exit Interview Analysis Report")
                        st.text_area("Report Content", report_text, height=400)
                        
                        # Download button
                        st.download_button(
                            label="Download Report",
                            data=report_text,
                            file_name=f"Exit_Interview_Report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"Error: {report_result.get('message', 'Unknown error')}")
            
          
        else:
            st.error(f"Error loading metadata: {metadata.get('message', 'Unknown error')}")
    else:
        st.error(f"❌ API Connection Failed: {health_check.get('message', 'Could not connect to API')}")
        st.info("Please check that the API server is running and the URL is correct.")

if __name__ == "__main__":
    main()