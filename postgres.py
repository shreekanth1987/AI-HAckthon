import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# --- Configuration ---
# IMPORTANT: When creating a new database, you must connect to an existing database first,
# typically 'postgres' or the default database created by your installation.
HOST = "localhost"
USER = "postgres" # Replace with your PostgreSQL username
PASSWORD = "test123" # Replace with your PostgreSQL password
PORT = 5432
DEFAULT_DB = "postgres" # The default database to connect to initially

# --- New database and schema details ---
NEW_DATABASE_NAME = "pyspark_analytics_db"
NEW_SCHEMA_NAME = "medallion_silver"

def connect_to_postgres(dbname):
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=HOST,
            database=dbname,
            user=USER,
            password=PASSWORD,
            port=PORT
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database '{dbname}': {e}")
        return None

def create_database(db_name):
    """
    Creates a new database in PostgreSQL.
    Requires autocommit mode and connection to the default database (e.g., 'postgres').
    """
    conn = None
    try:
        # 1. Connect to the default database
        conn = connect_to_postgres(DEFAULT_DB)
        if conn is None:
            return False

        # 2. Set isolation level to AUTOCOMMIT
        # DDL statements like CREATE DATABASE cannot run inside a transaction block.
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # 3. Construct the CREATE DATABASE query safely
        # Use psycopg2.sql.Identifier for database name to prevent SQL injection
        print(f"Attempting to create database: {db_name}...")
        create_db_query = sql.SQL("CREATE DATABASE {}").format(
            sql.Identifier(db_name)
        )
        
        # 4. Execute the query
        cursor.execute(create_db_query)
        print(f"SUCCESS: Database '{db_name}' created.")
        return True

    except psycopg2.ProgrammingError as e:
        # Handle case where database already exists
        if f'database "{db_name}" already exists' in str(e):
            print(f"INFO: Database '{db_name}' already exists. Continuing.")
            return True
        print(f"Error creating database: {e}")
        return False
        
    except Exception as e:
        print(f"An unexpected error occurred during database creation: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def create_schema(db_name, schema_name):
    """
    Creates a new schema within a specified database.
    Does not require autocommit since it runs inside a standard transaction.
    """
    conn = None
    try:
        # 1. Connect to the newly created database
        conn = connect_to_postgres(db_name)
        if conn is None:
            return

        cursor = conn.cursor()

        # 2. Construct the CREATE SCHEMA query safely
        # Use IF NOT EXISTS to prevent errors if the schema already exists
        print(f"Attempting to create schema: {schema_name} in {db_name}...")
        create_schema_query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
            sql.Identifier(schema_name)
        )
        
        # 3. Execute the query
        cursor.execute(create_schema_query)
        conn.commit() # Commit the transaction
        print(f"SUCCESS: Schema '{schema_name}' created in database '{db_name}'.")

    except Exception as e:
        print(f"Error creating schema: {e}")
        if conn:
            conn.rollback() # Rollback on error
        
    finally:
        if conn:
            conn.close()

# --- Execution ---
if __name__ == "__main__":
    
    # 1. Create the database
    db_created = create_database(NEW_DATABASE_NAME)

    # 2. If database creation was successful (or it already existed), create the schema inside it
    if db_created:
        create_schema(NEW_DATABASE_NAME, NEW_SCHEMA_NAME)
    
    print("\n--- Script Finished ---")
    print(f"To connect to the new database from the command line:")
    print(f"psql -U {USER} -d {NEW_DATABASE_NAME}")

    # You can now proceed to run PySpark applications against the new database and schema.