from app import create_app
from database import db

app = create_app()

with app.app_context():
    exams = db.get_all_examinations()

    counts = {}
    for exam in exams:
        status = exam.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    print("\n=== EXAMINATION STATUS SUMMARY ===")
    print(f"Total examinations: {len(exams)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print("\n=== RECENT EXAMINATIONS ===")
    for exam in exams[-20:]:
        print(
            f"ID={exam.get('id')} | "
            f"label={exam.get('patient_label')} | "
            f"status={exam.get('status')}"
        )
