# GitHub Setup Instructions

Follow these steps to push your neuroscience-modeling project to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account
- GitHub CLI (optional) or Personal Access Token

## Option 1: Using GitHub CLI (Easiest)

### Step 1: Install GitHub CLI (if not installed)
```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# Windows
winget install --id GitHub.cli
```

### Step 2: Authenticate
```bash
gh auth login
```

### Step 3: Navigate to your project
```bash
cd /path/to/neuroscience-modeling
```

### Step 4: Initialize and push
```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Neuroscience Modeling Framework"

# Create repository on GitHub and push
gh repo create neuroscience-modeling --public --source=. --push
```

## Option 2: Using Git and GitHub Web Interface

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `neuroscience-modeling`
3. Description: "A Docker-based neuroscience modeling framework with clean architecture"
4. Choose Public
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Push your local repository

```bash
# Navigate to your project
cd /path/to/neuroscience-modeling

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Neuroscience Modeling Framework"

# Add remote (replace 'mohoog10' with your username)
git remote add origin https://github.com/mohoog10/neuroscience-modeling.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Option 3: Using SSH

### Step 1: Set up SSH key (if not done)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add SSH key
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

Then add the key to GitHub:
1. Go to GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste your key and save

### Step 2: Push with SSH

```bash
# Navigate to your project
cd /path/to/neuroscience-modeling

# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Neuroscience Modeling Framework"

# Add remote with SSH
git remote add origin git@github.com:mohoog10/neuroscience-modeling.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Verification

After pushing, verify your repository:

1. Visit: https://github.com/mohoog10/neuroscience-modeling
2. Check that all files are present
3. Verify README.md displays correctly
4. Test cloning: `git clone https://github.com/mohoog10/neuroscience-modeling.git`

## Repository Settings (Recommended)

### Add Topics
Add these topics to your repository for better discoverability:
- python
- docker
- machine-learning
- neuroscience
- clean-architecture
- neural-networks
- docker-compose
- model-training

### Enable GitHub Actions (Optional)
You can add CI/CD later with a `.github/workflows/test.yml` file.

### Add Repository Description
"A Docker-based neuroscience modeling framework with clean architecture for model training, validation, and testing"

### Enable Discussions (Optional)
Settings → Features → Enable Discussions

## Updating Your Repository

After making changes:

```bash
# Check status
git status

# Add changed files
git add .

# Or add specific files
git add src/models/model3.py

# Commit with descriptive message
git commit -m "Add Model3 implementation"

# Push changes
git push
```

## Common Git Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature/new-model

# Switch branches
git checkout main

# Merge branch
git merge feature/new-model

# Pull latest changes
git pull origin main

# View remotes
git remote -v

# View differences
git diff
```

## Troubleshooting

### Issue: Permission denied
**Solution**: Check your authentication method (HTTPS token or SSH key)

### Issue: Repository already exists
**Solution**: Use `git pull origin main` first, then push

### Issue: Merge conflicts
**Solution**: 
```bash
git pull origin main
# Resolve conflicts in files
git add .
git commit -m "Resolve merge conflicts"
git push
```

### Issue: Large files warning
**Solution**: Add files to .gitignore or use Git LFS for large files

## Best Practices

1. ✅ Commit frequently with descriptive messages
2. ✅ Use branches for new features
3. ✅ Keep commits focused (one feature per commit)
4. ✅ Write meaningful commit messages
5. ✅ Pull before pushing if working with others
6. ✅ Review changes before committing
7. ✅ Don't commit sensitive information
8. ✅ Use .gitignore effectively

## Example Workflow

```bash
# Create feature branch
git checkout -b feature/add-model3

# Make changes to your code
# ... edit files ...

# Check what changed
git status
git diff

# Stage changes
git add src/models/model3.py
git add main.py

# Commit
git commit -m "Add Model3 with transformer architecture"

# Push feature branch
git push -u origin feature/add-model3

# Create pull request on GitHub
# After review and approval, merge on GitHub

# Switch back to main and update
git checkout main
git pull origin main

# Delete feature branch
git branch -d feature/add-model3
```

## Additional Resources

- [GitHub Documentation](https://docs.github.com)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [Pro Git Book](https://git-scm.com/book/en/v2)

## Quick Reference

```bash
# Clone a repository
git clone <url>

# Initialize repository
git init

# Stage files
git add <file>
git add .

# Commit
git commit -m "message"

# Push
git push origin main

# Pull
git pull origin main

# Check status
git status

# View history
git log

# Create branch
git branch <name>

# Switch branch
git checkout <name>

# Merge branch
git merge <name>
```

---

**Good luck with your GitHub repository! 🚀**

Your neuroscience-modeling framework is ready to share with the world!
