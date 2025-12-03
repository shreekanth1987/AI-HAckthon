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

def setup_and_insert_data(conn):
    """Creates the table and inserts multiple records."""
    cursor = conn.cursor()

    try:
        # 1. Define the fully qualified table name
        full_table_name = sql.SQL("{}.{}").format(
            sql.Identifier(SCHEMA_NAME),
            sql.Identifier(TABLE_NAME)
        )

        # 2. Create Table (using IF NOT EXISTS to prevent errors on repeated runs)
        print(f"Creating table {SCHEMA_NAME}.{TABLE_NAME}...")
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                flight_id VARCHAR(10) NOT NULL,
                departure_airport VARCHAR(3),
                arrival_airport VARCHAR(3),
                is_valid_schema BOOLEAN,
                ingestion_timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            );
        """).format(full_table_name)
        cursor.execute(create_table_query)
        print("Table creation/check successful.")

        # 3. Data to be inserted (list of tuples)
        records_to_insert = [
            ('AA101', 'JFK', 'LAX', True),
            ('UA205', 'ORD', 'MIA', True),
            ('DL330', 'ATL', 'SFO', False), # Example of data failing quality check
            ('SW477', 'DAL', 'PHX', True)
        ]

        # 4. Insert Query
        insert_query = sql.SQL("""
            INSERT INTO {} (flight_id, departure_airport, arrival_airport, is_valid_schema)
            VALUES (%s, %s, %s, %s);
        """).format(full_table_name)

        # 5. Execute batch insert
        print(f"Inserting {len(records_to_insert)} records...")
        cursor.executemany(insert_query, records_to_insert)

        # 6. Commit the changes to make them permanent
        conn.commit()
        print("SUCCESS: Data insertion complete.")

    except Exception as e:
        print(f"Insertion failed: {e}")
        conn.rollback()
    finally:
        cursor.close()

if __name__ == "__main__":
    db_conn = connect_to_db()
    if db_conn:
        setup_and_insert_data(db_conn)
        db_conn.close()