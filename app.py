# app.py - Updated to use database instead of file uploads
from flask import Flask, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import os
import json
import tempfile
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sqlalchemy import create_engine, text
import psycopg2

from dotenv import load_dotenv
import os
from exit_analyzer import ExitInterviewAnalyzer

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

load_dotenv()

DB_URI = os.getenv("DB_URI")



# Initialize our analyzer with LLM always enabled
groq_api_key = os.getenv("GROQ_API_KEY")
analyzer = ExitInterviewAnalyzer( llm_api_key=groq_api_key)

# Create database connection pool
engine = create_engine(DB_URI)

def get_data_from_db(department=None, role=None, tenure=None):
    """
    Query data from database with optional filters
    """
    try:
        # Start with base query
        query = "SELECT * FROM exit_interviews WHERE 1=1"
        params = {}
        
        # Add filters if specified
        if department and department != 'All':
            query += " AND department = :department"
            params['department'] = department
            
        if role and role != 'All':
            query += " AND role = :role"
            params['role'] = role
            
        if tenure and tenure != 'All':
            query += " AND tenure = :tenure"
            params['tenure'] = float(tenure)
            
        # Execute query with parameters
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    except Exception as e:
        print(f"Database error: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def get_db_metadata():
    """
    Get metadata about available departments, roles, and tenure categories
    """
    try:
        with engine.connect() as conn:
            # Get unique departments
            dept_result = conn.execute(text("SELECT DISTINCT department FROM exit_interviews ORDER BY department"))
            departments = [row[0] for row in dept_result]
            
            # Get unique roles
            role_result = conn.execute(text("SELECT DISTINCT role FROM exit_interviews ORDER BY role"))
            roles = [row[0] for row in role_result]
            
            # Get unique tenure values
            tenure_result = conn.execute(text("SELECT DISTINCT tenure FROM exit_interviews ORDER BY tenure"))
            tenures = [str(row[0]) for row in tenure_result]
            
        return {
            'departments': departments,
            'roles': roles,
            'tenure_categories': tenures
        }
    except Exception as e:
        print(f"Error fetching metadata: {str(e)}")
        return {'departments': [], 'roles': [], 'tenure_categories': []}

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get metadata about available filter options"""
    try:
        metadata = get_db_metadata()
        metadata['status'] = 'success'
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze the data with optional filters"""
    try:
        # Get filter parameters from request
        data = request.json
        department = data.get('department')
        role = data.get('role')
        tenure = data.get('tenure')
        
        # Apply filters if they're not empty/null
        department = department if department and department != 'All' else None
        role = role if role and role != 'All' else None
        tenure = tenure if tenure and tenure != 'All' else None
        
        # Create filter description for reports
        filter_desc = "All Data"
        if department or role or tenure:
            filter_parts = []
            if department:
                filter_parts.append(f"Department: {department}")
            if role:
                filter_parts.append(f"Role: {role}")
            if tenure:
                filter_parts.append(f"Tenure: {tenure}")
            filter_desc = ", ".join(filter_parts)
        
        # Get data from database
        df = get_data_from_db(department, role, tenure)
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found matching the specified filters'})
        
        # Get insights
        insights = analyzer.filter_insights(df, department, role, tenure)
        
        if "error" in insights:
            return jsonify({'status': 'error', 'message': insights['error']})
        
        # Generate heatmap
        heatmap_base64 = generate_heatmap(insights["category_percentages"])
        
        # Generate LLM report
        llm_report = analyzer.generate_llm_report(insights, filter_desc)
        
        # Return results
        return jsonify({
            'status': 'success',
            'insights': {
                'record_count': insights['record_count'],
                'category_percentages': {k: round(v, 1) for k, v in insights['category_percentages'].items()},
                'key_themes': insights['key_themes'],
                'filter_description': filter_desc,
                'heatmap': heatmap_base64,
                'llm_report': llm_report
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error analyzing data: {str(e)}'})

@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate a full report"""
    try:
        # Get filter parameters
        data = request.json
        department = data.get('department')
        role = data.get('role')
        tenure = data.get('tenure')
        
        # Apply filters if they're not empty/null
        department = department if department and department != 'All' else None
        role = role if role and role != 'All' else None
        tenure = tenure if tenure and tenure != 'All' else None
        
        # Get data from database
        df = get_data_from_db(department, role, tenure)
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found matching the specified filters'})
            
        # Always use LLM for enhanced insights
        report_text = analyzer.generate_summary_report(df, use_llm=True)
        
        # Return the report
        return jsonify({
            'status': 'success',
            'report': report_text
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error generating report: {str(e)}'})

@app.route('/api/download-report', methods=['POST'])
def download_report():
    """Generate and download a full report as a text file"""
    report_text = request.json.get('report_text')
    if not report_text:
        return jsonify({'status': 'error', 'message': 'No report text provided'})
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.write(report_text.encode('utf-8'))
    temp_file.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(temp_file.name, 
                     mimetype='text/plain',
                     as_attachment=True,
                     download_name=f'Exit_Interview_Report_{timestamp}.txt')

def generate_heatmap(category_percentages):
    """Generate a heatmap visualization of exit reasons"""
    # Filter out categories with 0%
    filtered_categories = {k: v for k, v in category_percentages.items() if v > 0}
    
    # Sort by percentage (descending)
    sorted_categories = sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    categories = []
    percentages = []
    
    for category, percentage in sorted_categories:
        # Format category name for display
        display_category = ' '.join(word.capitalize() for word in category.split('_'))
        categories.append(display_category)
        percentages.append(percentage)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create a custom blue colormap
    colors = [(0.9, 0.95, 1), (0, 0.3, 0.7)]  # Light blue to dark blue
    blue_cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=100)
    
    # Create horizontal bar chart
    bars = plt.barh(categories, percentages, color=plt.cm.Blues(0.6))
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{percentages[i]:.1f}%', va='center')
    
    plt.xlabel('Percentage of Exit Interviews')
    plt.title('Reasons for Leaving')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Exit Interview Analysis API is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)