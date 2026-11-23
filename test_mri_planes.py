from app import create_app

app = create_app()
client = app.test_client()

exam_id = 223

for view in ("axial", "coronal", "sagittal"):
    url = f"/analysis/{exam_id}/slice?modality=flair&view={view}&region=ALL&source=predicted"
    response = client.get(url)

    print("\n", view.upper())
    print("HTTP:", response.status_code)

    try:
        data = response.get_json()
        if isinstance(data, dict):
            print("Response keys:", list(data.keys()))
            image = data.get("image")
            print("Image rows:", len(image) if image else 0)
            print("Image columns:", len(image[0]) if image and image[0] else 0)
            if response.status_code != 200:
                print("Error:", data.get("error"))
        else:
            print("Response:", data)
    except Exception:
        print(response.get_data(as_text=True)[:500])
