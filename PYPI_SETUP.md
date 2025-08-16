# PyPI Publishing Setup Guide

This document explains how to set up automated PyPI publishing for the Turkish Tokenizer with Gemma Model package.

## 🚀 Quick Setup Checklist

- [x] ✅ GitHub Actions workflow created (`.github/workflows/publish.yml`)
- [x] ✅ Package configuration files created (`pyproject.toml`, `setup.py`)
- [x] ✅ Package structure organized (`tr_tokenizer_gemma/`)
- [x] ✅ README updated with installation instructions
- [x] ✅ Basic tests created (`tests/`)
- [x] ✅ License file added (`LICENSE`)

## 📋 Steps to Complete Setup

### 1. Update Package Metadata

Edit the following files with your actual information:

**`pyproject.toml`** and **`setup.py`**:

```toml
authors = [
    {name = "Your Actual Name", email = "your.actual.email@example.com"}
]
```

**`pyproject.toml`** URLs section:

```toml
[project.urls]
Homepage = "https://github.com/yourusername/tokenizer"
Repository = "https://github.com/yourusername/tokenizer"
Issues = "https://github.com/yourusername/tokenizer/issues"
```

### 2. Set Up PyPI Account and Trusted Publishing

1. **Create PyPI Account**: Go to https://pypi.org and create an account if you don't have one

2. **Set Up Trusted Publishing** (Recommended):

   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in the form:
     - **Owner**: `malibayram` (your GitHub username)
     - **Repository name**: `tokenizer`
     - **Workflow name**: `publish.yml`
     - **Environment name**: `release` (optional but recommended)

3. **Create GitHub Environment** (Recommended for security):
   - Go to your GitHub repository
   - Navigate to Settings → Environments
   - Click "New environment"
   - Name it `release`
   - Add protection rules (require reviews, restrict to main branch, etc.)

### 3. Alternative: Manual Token Setup

If you prefer not to use trusted publishing:

1. **Generate PyPI API Token**:

   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with scope for the specific project
   - Copy the token (starts with `pypi-`)

2. **Add Token to GitHub Secrets**:

   - Go to your GitHub repository
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI token

3. **Update Workflow** (if using manual tokens):
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

### 4. Test the Setup

1. **Local Build Test**:

   ```bash
   cd /path/to/your/tokenizer
   python -m build
   python -m twine check dist/*
   ```

2. **Test Installation**:
   ```bash
   pip install dist/*.whl
   python -c "from tr_tokenizer_gemma import TRTokenizer; print('Success!')"
   ```

### 5. Trigger First Release

#### Option A: Tag-based Release (Recommended)

```bash
git tag v0.1.0
git push origin v0.1.0
```

#### Option B: Push to Main (Will publish to TestPyPI)

```bash
git add .
git commit -m "feat: setup PyPI publishing"
git push origin main
```

## 🔧 Workflow Behavior

The GitHub Actions workflow:

1. **Runs tests** on Python 3.8-3.12 for every push/PR
2. **Builds package** after tests pass
3. **Publishes to PyPI** when:
   - A tag starting with `v` is pushed (e.g., `v0.1.0`) → **Production PyPI**
   - Code is pushed to `main` branch → **TestPyPI** (for testing)

## 📦 Package Structure

```
tokenizer/
├── .github/workflows/publish.yml    # GitHub Actions workflow
├── tr_tokenizer_gemma/              # Main package
│   ├── __init__.py                  # Package exports
│   ├── tr_tokenizer.py             # Tokenizer implementation
│   ├── gemma_model.py              # Gemma model implementation
│   ├── tr_decoder.py               # Decoder implementation
│   └── *.json                      # Tokenizer data files
├── tests/                           # Test files
├── pyproject.toml                   # Modern Python packaging
├── setup.py                         # Legacy Python packaging
├── MANIFEST.in                      # Include/exclude files
├── requirements.txt                 # Dependencies
├── LICENSE                          # MIT License
└── README.md                        # Updated with installation
```

## 🛠 Customization

### Version Management

Update version in both:

- `pyproject.toml`: `version = "0.1.1"`
- `setup.py`: `version="0.1.1"`
- `tr_tokenizer_gemma/__init__.py`: `__version__ = "0.1.1"`

### Add Dependencies

Edit `pyproject.toml`:

```toml
dependencies = [
    "torch>=1.9.0",
    "numpy>=1.21.0",
    "your-new-dependency>=1.0.0",
]
```

### Modify Publishing Conditions

Edit `.github/workflows/publish.yml` to change when publishing happens:

```yaml
# Publish only on tags
if: startsWith(github.ref, 'refs/tags/v')

# Or publish on specific branches
if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/release'
```

## 🎯 Ready to Publish!

Once you've completed the setup:

1. Update your actual information in the configuration files
2. Set up PyPI trusted publishing or add API token
3. Create a release tag: `git tag v0.1.0 && git push origin v0.1.0`
4. Watch the GitHub Actions workflow publish your package!

Your package will be available as:

```bash
pip install turkish-tokenizer
```

## 📚 Additional Resources

- [PyPI Trusted Publishers Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Python Packaging User Guide](https://packaging.python.org/)
