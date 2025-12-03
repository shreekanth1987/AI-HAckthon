import psycopg2
from psycopg2 import sql

# --- Configuration (Copied from postgres_admin.py) ---
HOST = "localhost"
USER = "postgres" 
PASSWORD = "postgres" # Ensure this is your correct password
PORT = 5432
DATABASE_NAME = "AIFORCEONE"
SCHEMA_NAME = "medallion_silver"
TABLE_NAME = "flight_quality_check"

def connect_to_db():
    """Establishes a connection to the target database."""
    try:
        conn = psycopg2.connect(
            host=HOST,
            database=DATABASE_NAME,
            user=USER,
            password=PASSWORD,
            port=PORT
        )
        print(f"Connection established to '{DATABASE_NAME}'.")
        return conn
    except Exception as e:
        print(f"Error connecting to database '{DATABASE_NAME}': {e}")
        return None

def retrieve_all_data(conn):
    """Executes a SELECT query and prints all results."""
    cursor = conn.cursor()

    try:
        # 1. Define the fully qualified table name
        full_table_name = sql.SQL("{}.{}").format(
            sql.Identifier(SCHEMA_NAME),
            sql.Identifier(TABLE_NAME)
        )

        # 2. Select Query
        select_query = sql.SQL("SELECT * FROM {};").format(full_table_name)
        
        # 3. Execute the query
        print(f"Executing query: SELECT * FROM {SCHEMA_NAME}.{TABLE_NAME}...")
        cursor.execute(select_query)

        # 4. Fetch all results
        records = cursor.fetchall()
        
        # 5. Get column names for headers
        column_names = [desc[0] for desc in cursor.description]

        # 6. Print the results neatly
        print("-" * 50)
        print(f"Retrieved {len(records)} records:")
        print("-" * 50)
        print(column_names)
        print("-" * 50)
        
        for row in records:
            # Note: The timestamp column will appear as a Python datetime object
            print(row)
        print("-" * 50)

    except Exception as e:
        print(f"Retrieval failed: {e}")
    finally:
        cursor.close()

if __name__ == "__main__":
    db_conn = connect_to_db()
    if db_conn:
        retrieve_all_data(db_conn)
        db_conn.close()