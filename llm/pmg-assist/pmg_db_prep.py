from os import environ
from dotenv import load_dotenv
from pmg_db import init_db

def main():
    """Main function to initialize the database."""
    environ["RUN_TIMEZONE_CHECK"] = "0"
    load_dotenv()
    print("Initializing database...")
    init_db()

if __name__ == "__main__":
    main()
