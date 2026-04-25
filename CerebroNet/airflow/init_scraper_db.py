import sqlite3
import os

def init_db():
    # Automatically resolves to the same directory as this script (the airflow folder)
    db_path = os.path.join(os.path.dirname(__file__), "scraper.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create pages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        html_path TEXT,
        js_path TEXT,
        images_count INTEGER,
        status TEXT NOT NULL,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
