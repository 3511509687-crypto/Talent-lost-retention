# GitHub Upload Guide

Run the commands from the project root after cloning or opening this repository locally.

## 1. Review The Working Tree

```powershell
cd path/to/your/project
git status --short --ignored
```

Check that source files, docs, tests, curated datasets, and configuration are included, while caches and generated reports are ignored.

## 2. Remove Already-Tracked Generated Files From Git Index

`.gitignore` prevents new ignored files from being added, but files already tracked by Git need to be removed from the index once.

```powershell
git rm -r --cached __pycache__ models/__pycache__ services/__pycache__ tests/__pycache__
git rm -r --cached .pycache_check model_cache model_outputs pip_tmp python_cuda_packages runtime_tmp
git rm -r --cached legacy_versions
git rm --cached result预测结果.xlsx
```

If one of these paths is not tracked, Git may report that it did not match any files. That is fine.

## 3. Stage The Current Source Tree

```powershell
git add .gitignore README.md app.py web_latest_app.py run_app.bat
git add models services static tests tools docs uploads
git status --short
```

Before committing, make sure large generated reports, cache folders, local dependency folders, logs, and PPT files are not staged.

## 4. Commit

```powershell
git commit -m "Prepare attrition predictor project for GitHub"
```

## 5. Create A GitHub Repository

Create a new empty repository on GitHub. Do not initialize it with a README if this local repository already has one.

Then connect the remote:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
```

If `origin` already exists:

```powershell
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```

## 6. Push

```powershell
git branch -M main
git push -u origin main
```

## 7. If The Push Is Too Large

Large local folders such as `python_cuda_packages/`, `model_cache/`, generated Excel reports, or PPT workspaces should not be committed. Run:

```powershell
git status --short
```

Then remove any accidentally staged generated files:

```powershell
git restore --staged <path>
```

After that, commit again.
