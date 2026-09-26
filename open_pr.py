from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
repo = "MuXodious/Heretic-Models-Explorer"

ops = [
    CommitOperationAdd("src/streamlit_app.py", "src/streamlit_app.py"),
    CommitOperationAdd("app.py", "app.py"),
    CommitOperationAdd("src/popularity_store.py", "src/popularity_store.py"),
    CommitOperationAdd("data/popularity.json", "data/popularity.json"),
    CommitOperationAdd("data/.gitkeep", "data/.gitkeep"),
    CommitOperationAdd("Dockerfile", "Dockerfile"),
    CommitOperationAdd("docker-compose.yml", "docker-compose.yml"),
    CommitOperationAdd(".dockerignore", ".dockerignore"),
]

info = api.create_commit(
    repo_id=repo, repo_type="space",
    operations=ops,
    commit_message="feat: date pre-filter + unknowns scope, popularity store, honest compare",
    commit_description="Date pre-filter before parsing, Include/Hide/Only unknowns, popularity in data/popularity.json, no-trophy compare, reset fix. Tested: docker compose up --build.",
    create_pr=True,
)
print(info.pr_url or "check Discussions tab")
