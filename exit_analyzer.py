import os
import re
import json
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import AutoTokenizer
from datetime import datetime
from groq import Groq

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load required models
hf_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    # Create a simple fallback class to mimic spaCy functionality
    class SimpleNLP:
        def __call__(self, text):
            return SimpleDoc(text)
    
    class SimpleDoc:
        def __init__(self, text):
            self.text = text
            self.ents = []
    
    nlp = SimpleNLP()
    SPACY_AVAILABLE = False

class ExitInterviewAnalyzer:
    def __init__(self, llm_api_key=None):
        # Initialize tokenizer and NLP tools
        self.tokenizer = hf_tokenizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # TF-IDF vectorizer for topic modeling
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Add domain-specific stopwords
        self.stop_words.update(['company', 'organization', 'employee', 'work'])
        
        # Predefined categories of reasons for leaving
        self.reason_categories = {
            'career_growth': ['promotion', 'advancement', 'growth', 'opportunity', 'career', 'develop', 'progress'],
            'compensation': ['salary', 'pay', 'compensation', 'bonus', 'wage', 'benefit', 'money'],
            'work_life_balance': ['balance', 'hour', 'overtime', 'weekend', 'flexibility', 'burnout', 'stress'],
            'management': ['manager', 'supervisor', 'leadership', 'micromanage', 'boss'],
            'culture': ['culture', 'environment', 'toxic', 'inclusive', 'diversity', 'harassment', 'morale'],
            'relocation': ['move', 'relocate', 'location', 'commute', 'distance'],
            'job_satisfaction': ['bored', 'challenging', 'fulfilling', 'meaningful', 'satisfaction', 'passion'],
            'workload': ['overworked', 'understaffed', 'pressure', 'deadline', 'overwhelming'],
            'recognition': ['appreciate', 'recognized', 'valued', 'acknowledged', 'reward'],
            'company_direction': ['strategy', 'vision', 'direction', 'leadership', 'future']
        }
        
        # LLM setup
        self.llm_api_key = llm_api_key or os.environ.get("GROQ_API_KEY")
        self.use_llm = bool(self.llm_api_key)
        if self.use_llm:
            try:
                self.llm_client = Groq(api_key=self.llm_api_key)
                self.llm_model = "gemma2-9b-it"
            except Exception:
                self.use_llm = False

    def preprocess_text(self, text):
        """Clean and preprocess text data using Hugging Face"""
        # Handle empty or non-string input
        if not isinstance(text, str) or not text.strip():
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize using Hugging Face tokenizer
        tokens = self.tokenizer.tokenize(text)

        # Remove stopwords and short tokens, then lemmatize
        clean_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        # Remove "##" from subwords 
        return ' '.join([token.replace('##', '') for token in clean_tokens])
    
    def extract_entities(self, text):
        """Extract named entities in the text"""      
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities
    
    def identify_reason_categories(self, processed_text):
        """Identify which predefined categories the text falls into"""
        matches = {}
        
        for category, keywords in self.reason_categories.items():
            count = 0
            for keyword in keywords:
                if keyword in processed_text:
                    count += 1
            
            if count > 0:
                matches[category] = count
                
        return matches
    
    def extract_key_themes(self, texts):
        """Extract key themes using TF-IDF and clustering"""
        # Handle empty input
        if not texts:
            return [["No data available"]]
            
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        processed_texts = [text for text in processed_texts if text]
        
        # If no valid texts remain, return placeholder
        if not processed_texts:
            return [["No processable data available"]]
            
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Determine number of topics based on data size
        n_topics = min(5, len(processed_texts))
        
        # If only one document, extract top terms directly
        if n_topics == 1:
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
            top_words = [feature_names[i] for i in tfidf_sorting[:10] if i < len(feature_names)]
            return [top_words]
        
        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # Get top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        themes = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            themes.append(top_words)
            
        return themes
    
    def categorize_all_interviews(self, interviews_df):
        """Categorize all interviews and calculate percentages"""
        # Handle empty dataframe
        if interviews_df.empty:
            return {category: 0 for category in self.reason_categories.keys()}
            
        # Add a new column with preprocessed text
        interviews_df['processed_text'] = interviews_df['interview_text'].apply(self.preprocess_text)
        
        # Identify reason categories for each interview
        interviews_df['reason_categories'] = interviews_df['processed_text'].apply(self.identify_reason_categories)
        
        # Count occurrences of each category
        category_counts = {category: 0 for category in self.reason_categories.keys()}
        
        for _, row in interviews_df.iterrows():
            for category in row['reason_categories'].keys():
                category_counts[category] += 1
        
        # Calculate percentages
        total_interviews = len(interviews_df)
        category_percentages = {category: (count / total_interviews) * 100 
                               for category, count in category_counts.items()}
        
        return category_percentages
    
    def filter_insights(self, interviews_df, department=None, role=None, tenure=None):
        """Filter insights by department, role, and tenure"""
        # Handle empty dataframe
        if interviews_df.empty:
            return {"error": "No data available"}
            
        filtered_df = interviews_df.copy()
        
        if department:
            filtered_df = filtered_df[filtered_df['department'] == department]
            
        if role:
            filtered_df = filtered_df[filtered_df['role'] == role]
            
        if tenure:
            filtered_df = filtered_df[filtered_df['tenure'] == tenure]
            
        # If no records match the filter
        if len(filtered_df) == 0:
            return {"error": "No records match the specified filters"}
            
        # Get category percentages for the filtered data
        category_percentages = self.categorize_all_interviews(filtered_df)
        
        # Extract key themes from the filtered data
        key_themes = self.extract_key_themes(filtered_df['interview_text'].tolist())
        
        return {
            "category_percentages": category_percentages,
            "key_themes": key_themes,
            "record_count": len(filtered_df),
            "interviews": filtered_df[['department', 'role', 'tenure', 'interview_text']].to_dict('records')
        }

    def generate_llm_report(self, insights_data, filter_description="overall"):
        """Generate an insightful report using Groq LLM"""
        if not self.use_llm:
            return "LLM reporting disabled. Please enable and provide Groq API key in settings."
        
        # Format the insights data for the LLM
        percentages = insights_data["category_percentages"]
        themes = insights_data["key_themes"]
        record_count = insights_data["record_count"]
        
        # Sample of interview texts (up to 5)
        sample_interviews = []
        if "interviews" in insights_data:
            sample_size = min(5, len(insights_data["interviews"]))
            sample_interviews = [interview["interview_text"] for interview in insights_data["interviews"][:sample_size]]
        
        # Create prompt for LLM
        prompt = f"""
        You are an advanced HR analytics expert. Based on the exit interview data provided, generate a comprehensive, 
        insightful report that identifies patterns, provides actionable recommendations, and highlights key issues.
        
        DATA SUMMARY:
        - Filter applied: {filter_description}
        - Number of exit interviews analyzed: {record_count}
        
        REASON CATEGORIES (percentage of interviews mentioning each):
        {json.dumps({k: f"{v:.1f}%" for k, v in percentages.items() if v > 0}, indent=2)}
        
        KEY THEMES IDENTIFIED (words associated with each theme):
        {json.dumps([', '.join(theme[:5]) for theme in themes], indent=2)}
        
        SAMPLE INTERVIEW EXCERPTS:
        {json.dumps(sample_interviews, indent=2)}
        
        Please analyze this data and provide:
        1. Executive Summary: 2-3 sentences summarizing the key findings
        2. Detailed Analysis: Break down the main reasons people are leaving
        3. Patterns and Trends: Identify any notable patterns in the data
        4. Action Recommendations: 3-5 specific, actionable recommendations for HR and management
        5. Risk Assessment: Identify potential risks if these issues are not addressed
        
        Format the report in a professional, clear manner suitable for presentation to senior management.
        """
        
        try:
            # Use Groq instead of OpenAI
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,  # Use the model specified in constructor
                messages=[
                    {"role": "system", "content": "You are an HR analytics expert specializing in analyzing exit interview data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating LLM report: {str(e)}"
        
    def generate_summary_report(self, interviews_df, use_llm=True):
        """Generate a summary report of exit interview analysis"""
        # Handle empty dataframe
        if interviews_df.empty:
            return "No data available for analysis."
            
        # Overall category percentages
        overall_percentages = self.categorize_all_interviews(interviews_df)
        
        # Sort categories by percentage (descending)
        sorted_categories = sorted(overall_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Format the summary text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = [f"Exit Interview Analysis Summary (Generated on {timestamp})"]
        summary.append(f"Total interviews analyzed: {len(interviews_df)}")
        summary.append("\nKEY REASONS FOR LEAVING:")
        
        for category, percentage in sorted_categories:
            if percentage > 0:  # Only include non-zero categories
                # Format category name for display
                display_category = ' '.join(word.capitalize() for word in category.split('_'))
                summary.append(f"- {display_category}: mentioned in {percentage:.1f}% of exit interviews")
        
        # Extract overall themes
        themes = self.extract_key_themes(interviews_df['interview_text'].tolist())
        
        summary.append("\nKEY THEMES IDENTIFIED:")
        for i, theme_words in enumerate(themes, 1):
            summary.append(f"- Theme {i}: {', '.join(theme_words[:5])}")
            
        # Department breakdown
        departments = interviews_df['department'].unique()
        if len(departments) > 1:  # Only if multiple departments
            summary.append("\nDEPARTMENT BREAKDOWN:")
            
            for dept in departments:
                dept_df = interviews_df[interviews_df['department'] == dept]
                dept_count = len(dept_df)
                dept_percent = (dept_count / len(interviews_df)) * 100
                
                # Get top reason for this department
                try:
                    top_reason = self.get_top_reason(dept_df)
                    summary.append(f"- {dept}: {dept_count} exits ({dept_percent:.1f}%), Top reason: {top_reason}")
                except:
                    summary.append(f"- {dept}: {dept_count} exits ({dept_percent:.1f}%)")
        
        # Tenure breakdown if available
        if 'tenure' in interviews_df.columns:
            tenure_categories = interviews_df['tenure'].unique()
            if len(tenure_categories) > 1:
                summary.append("\nTENURE BREAKDOWN:")
                
                for tenure in sorted(tenure_categories):
                    tenure_df = interviews_df[interviews_df['tenure'] == tenure]
                    tenure_count = len(tenure_df)
                    tenure_percent = (tenure_count / len(interviews_df)) * 100
                    
                    # Get top reason for this tenure
                    try:
                        top_reason = self.get_top_reason(tenure_df)
                        summary.append(f"- {tenure}: {tenure_count} exits ({tenure_percent:.1f}%), Top reason: {top_reason}")
                    except:
                        summary.append(f"- {tenure}: {tenure_count} exits ({tenure_percent:.1f}%)")
        
        # Use LLM for enhanced insights if enabled
        if use_llm and self.use_llm:
            insights_data = {
                "category_percentages": overall_percentages,
                "key_themes": themes,
                "record_count": len(interviews_df),
                "interviews": interviews_df[['department', 'role', 'tenure', 'interview_text']].to_dict('records')
            }
            
            llm_report = self.generate_llm_report(insights_data)
            summary.append("\n\n" + "=" * 50 + "\n")
            summary.append("ENHANCED INSIGHTS (AI-GENERATED REPORT)")
            summary.append("=" * 50 + "\n")
            summary.append(llm_report)
            
        return "\n".join(summary)
    
    def get_top_reason(self, filtered_df):
        """Get top reason for a filtered dataset"""
        category_percentages = self.categorize_all_interviews(filtered_df)
        # Filter out zero percentages
        non_zero_categories = {k: v for k, v in category_percentages.items() if v > 0}
        
        if not non_zero_categories:
            return "Unknown"
            
        top_category = max(non_zero_categories.items(), key=lambda x: x[1])
        display_category = ' '.join(word.capitalize() for word in top_category[0].split('_'))
        return display_category