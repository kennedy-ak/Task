import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import sys
from sqlalchemy.types import VARCHAR, FLOAT, TEXT

# PostgreSQL connection string
DB_URI = "postgresql://rgt_database_dpt3_user:Uxy1HDb4QNzOuJA3LkJStUaCGwmfnbGk@dpg-cv4sfllumphs73fjgbtg-a.ohio-postgres.render.com/rgt_database_dpt3"

def create_table_and_import_data(csv_path):
    """
    Create the exit interviews table and import data from CSV
    """
    try:
        print(f"Reading CSV file from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"CSV data loaded successfully with {len(df)} rows")
        
        # Create SQLAlchemy engine for pandas to_sql
        engine = create_engine(DB_URI)
        
        # Create direct connection to execute SQL commands
        conn = psycopg2.connect(DB_URI)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop table if it exists
        cursor.execute("DROP TABLE IF EXISTS exit_interviews")
        
        # Create the table
        print("Creating table exit_interviews...")
        cursor.execute("""
        CREATE TABLE exit_interviews (
            id SERIAL PRIMARY KEY,
            employee_id VARCHAR(20) NOT NULL,
            department VARCHAR(100) NOT NULL,
            role VARCHAR(100) NOT NULL,
            tenure FLOAT NOT NULL,
            interview_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Define SQLAlchemy column types explicitly
        dtype = {
            'employee_id': VARCHAR(20),
            'department': VARCHAR(100),
            'role': VARCHAR(100),
            'tenure': FLOAT,
            'interview_text': TEXT
        }
        
        # Import data using pandas - without the id column that will be auto-generated
        print("Importing data from CSV to database...")
        # Ensure correct column order for insertion
        columns_order = ['employee_id', 'department', 'role', 'tenure', 'interview_text']
        
        # Explicitly tell pandas which columns to use and their order
        df[columns_order].to_sql(
            'exit_interviews', 
            engine, 
            if_exists='append', 
            index=False,
            dtype=dtype
        )
        
        # Verify data was imported
        cursor.execute("SELECT COUNT(*) FROM exit_interviews")
        row_count = cursor.fetchone()[0]
        print(f"Successfully imported {row_count} rows to the database")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_setup.py <path_to_csv_file>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    success = create_table_and_import_data(csv_path)
    
    if success:
        print("Database setup completed successfully!")
    else:
        print("Database setup failed!")
        sys.exit(1)