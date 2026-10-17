"""Load the seed knowledge documents (knowledge/documents/*.md) into the database.

Safe to run repeatedly: documents that already exist (matched by title) are skipped and
never overwritten. It is not run automatically when the app starts.

Run from the repository root, with DATABASE_URL set in .env:

    python -m scripts.seed_knowledge
"""

from app.db.database import Base, SessionLocal, engine
from app.services.knowledge import seed_documents
from app.services.knowledge_seed import load_seed_documents


def main() -> None:
    documents = load_seed_documents()  # fails before touching the database if a file is bad
    Base.metadata.create_all(bind=engine)  # same table setup as the app does at startup
    with SessionLocal() as db:
        result = seed_documents(db, documents)
        db.commit()
    print(f"Seed documents found: {len(documents)}")
    print(f"Inserted: {len(result.inserted)}")
    for title in result.inserted:
        print(f"  + {title}")
    print(f"Skipped (already present): {len(result.skipped)}")


if __name__ == "__main__":
    main()
