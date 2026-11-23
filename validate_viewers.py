from app import create_app

app = create_app()

for name in ("base.html", "results.html", "history.html"):
    app.jinja_env.get_template(name)
    print(name, "OK")

for rule in app.url_map.iter_rules():
    if "viewer" in rule.rule:
        print("ROUTE:", rule)

print("Flask validation passed.")
