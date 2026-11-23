from app import create_app
from urllib.parse import urlparse

app = create_app()
client = app.test_client()

with app.app_context():
    from app import db
    exams = db.get_all_examinations()
    exam = max(exams, key=lambda e: e["id"], default=None)

    if exam is None:
        print("No examinations found.")
    else:
        print("Testing examination:", exam["id"])
        response = client.get(f'/analysis/{exam["id"]}/results')
        print("Results page:", response.status_code)

        for path in (
            f'/viewer/2d/{exam["id"]}',
            f'/viewer/3d/{exam["id"]}'
        ):
            r = client.get(path, follow_redirects=False)
            print(path, "->", r.status_code, r.headers.get("Location"))
