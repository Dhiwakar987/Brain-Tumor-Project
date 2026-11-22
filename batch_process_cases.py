from pathlib import Path
from werkzeug.datastructures import FileStorage

from app import create_app
from services import file_service as fs
from database import db

LIMIT = 200
SKIP_CASES = {"BRATS_001", "BRATS_002", "BRATS_005"}
SOURCE_DIR = Path(r"C:\BraTS_Data\split_for_web")

app = create_app()


def main():
    svc = app.extensions["analysis_service"]

    with app.app_context():
        exams = db.get_all_examinations()
        existing_labels = {
            exam.get("patient_label")
            for exam in exams
            if exam.get("patient_label")
        }

        case_dirs = sorted(
            p for p in SOURCE_DIR.iterdir()
            if p.is_dir()
            and p.name not in SKIP_CASES
            and p.name not in existing_labels
            and all((p / f"{role}.nii.gz").is_file() for role in fs.MODALITIES)
        )[:LIMIT]

        print(f"Already recorded case labels: {len(existing_labels)}")
        print(f"New cases to process: {len(case_dirs)}")

        if not case_dirs:
            print("No unrecorded complete cases found.")
            return

        for index, case_dir in enumerate(case_dirs, start=1):
            opened = []
            try:
                files = {}
                for role in fs.MODALITIES:
                    path = case_dir / f"{role}.nii.gz"
                    stream = path.open("rb")
                    opened.append(stream)
                    files[role] = FileStorage(
                        stream=stream,
                        filename=path.name,
                    )

                print(f"\n[{index}/{len(case_dirs)}] Uploading {case_dir.name}...")
                result = svc.create_examination(files, case_dir.name)

                if not result.get("ok"):
                    print("UPLOAD FAILED:", result.get("errors"))
                    continue

                exam_id = result["id"]
                existing_labels.add(case_dir.name)
                print(f"Validated as examination #{exam_id}. Starting segmentation...")

                result = svc.start_segmentation(exam_id, wait=True)
                if not result.get("ok"):
                    print("SEGMENTATION COULD NOT START:", result)
                    continue

                exam = db.get_examination(exam_id)
                print(f"Finished examination #{exam_id}. Current status: {exam['status']}")

            except Exception as exc:
                print(f"ERROR for {case_dir.name}: {type(exc).__name__}: {exc}")
            finally:
                for stream in opened:
                    stream.close()


if __name__ == "__main__":
    main()