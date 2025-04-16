# 🏏 Innings-Score-Predictor - An End-to-End MLOps Project
This is an End to end Project implementing Machine Learning Operations to develop a production grade project which is used to predict the score of an innings based on certain features and user input values.

Using tools like DVC, MLflow, DagsHub, Flask, and AWS CodeBuild, CodeDeploy, CodePipeline, etc. It demonstrates a production-ready ML pipeline with experiment tracking, model versioning, and cloud integration.

---

## 🖥️ Demonstration of FlaskApp on Live Server 
https://github.com/user-attachments/assets/8d5775c5-c102-4c9d-a9e8-1b11d3437787

- Tested prediction of an Inning's End Score
- Validated if our logic holds or failed such as:
    - same bat-team and bowl-team
    - overs below 4.1 and above 20
    - wickets fell more than 10
---

## 🆕 Whats's New? : "CodePipeline Over Docker".

### ❌ Why Not Use Docker?
While Docker is a powerful tool for containerizing applications, it can sometimes introduce unnecessary complexity—especially for straightforward deployment pipelines or for beginners just getting into MLOps. Writing Dockerfiles, building images, and managing containers can slow down development and add infrastructure overhead that isn't always needed for smaller or more focused projects.

### ✅ Why CodePipeline?
Instead of Docker, this project uses AWS CodePipeline in combination with CodeBuild, CodeDeploy, S3, and EC2 to achieve a fully automated CI/CD setup. CodePipeline offers a visual, intuitive, and tightly integrated workflow that simplifies the deployment process. It requires no containerization and yet ensures robust automation, reproducibility, and scalability. For this project, it was a faster and more accessible way to implement production-ready ML deployment.

### 🔑 Key Takeaway
This approach shows that you don’t need Docker to build effective, real-world MLOps pipelines. Cloud-native tools like CodePipeline can handle the job efficiently while remaining user-friendly and maintainable. It lowers the barrier to entry for those new to MLOps and encourages quicker iterations and deployments without diving deep into infrastructure management.

### 🧠 Conclusion: A Simplified MLOps Approach
By choosing CodePipeline over Docker, this project introduces a fresh, simplified approach to MLOps deployment. It's ideal for solo developers, students, or teams looking for a scalable yet accessible solution. This project is a clear example of how modern cloud-native workflows can replace traditional container-based methods without compromising functionality or reliability.

---

## 🛠️ Project Setup & Execution

To **imitate or run this project end-to-end**, please refer to [`projectflow.txt`](./projectflow.txt)  
It contains **clear and precise, step-by-step directions** for setup, configuration, and running the application
using a modular approach and implementing an eased up CI-CD workflow.

This ensures consistency and helps you follow the same structure and flow used in this project.

---

## 📁 Project Setup Overview

### Step 1: Initialize Project

```Terminal
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
conda create -n ipl_project python=3.10 -y
conda activate ipl_project
```

### Step 2: Setup Folder Structure

```Terminal
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

- Rename `src.models` to `src.model`
- Add `data/` to `.gitignore`

```Terminal
git add .
git commit -m "Initial setup"
git push
```

---

## 🚀 MLflow Integration with DagsHub

### Step 3: Connect GitHub to DagsHub

- Go to: https://dagshub.com/dashboard
- Create new repo → Connect GitHub repo
- Copy the MLflow tracking code snippet

### Step 4: Install MLflow and DagsHub SDK

```Terminal
pip install dagshub 
pip install mlflow 
pip insyall ipykernel
pip install -r requirements.txt
```

### Step 5: Run Experiment Notebook

```Terminal
jupyter notebook
# Run experiment notebook and track via MLflow
```

```Terminal
git add .
git commit -m "Added experiment logs"
git push
```

---

## 📦 DVC Pipeline Setup

### Step 6: Initialize DVC

```Terminal
dvc init
mkdir local_s3
dvc remote add -d mylocal local_s3
```

### Step 7: Add Pipeline Code (inside src/)

- `logger/__init__.py`
- `data_ingestion.py`
- `data_preprocessing.py`
- `feature_engineering.py`
- `model_building.py`
- `model_evaluation.py`
- `register_model.py`

Add the following files to root:
- `dvc.yaml`
- `params.yaml`

```Terminal
dvc repro
dvc status
git add .
git commit -m "DVC pipeline added"
git push
```

---

## ☁️ AWS S3 Integration for DVC

### Step 8: Setup AWS

- Create IAM user with programmatic access
- Create S3 bucket (e.g., `ipl-project-bucket`)

### Step 9: Install AWS Tools

```Terminal
pip install "dvc[s3]" awscli
```

### Step 10: ✅ Environment Variable Setup

```Terminal
$env:AWS_ACCESS_KEY_ID="your_access_key_id"
$env:AWS_SECRET_ACCESS_KEY="your_secret_access_key"
$env:BUCKET_NAME="your_bucket_name"
```

### Step 11: Connect S3 as DVC Remote

```Terminal
dvc remote add -d myremote s3://your_bucket_name
dvc push
```

---

## 🌐 Flask App for Model Serving

### Step 12: Create Flask App Directory

```Terminal
mkdir flaskapp
cd flaskapp
# Add Flask app files like app.py, templates/, etc.
```

### Step 13: Run Flask App in your local system.

```Terminal
pip install flask
python app.py

```

---

## 🔐 DagsHub Token Setup

### Step 14: Generate Token

- Go to DagsHub > Repo > Settings > Access Tokens
- Click **Generate New Token**
- Name: `CAPSTONE_TEST`
- Copy the token

### Step 15: Set Token in Terminal

```Terminal
$env:CAPSTONE_TEST="your_token_here"
```

---

<!-- ## ✅ Environment Variable Setup

```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key_id"
$env:AWS_SECRET_ACCESS_KEY="your_secret_access_key"
$env:BUCKET_NAME="your_bucket_name" -->


---

## 📌 .gitignore Recommendations

```gitignore
data/
local_s3/
__pycache__/
*.log
*.env
mlruns/
```

---

## ✅ Final Checklist

- [x] Project structure initialized
- [x] MLflow tracking via DagsHub
- [x] DVC pipeline created and reproducible
- [x] AWS S3 remote configured
- [x] Flask app deployed for predictions
- [x] Secure environment variables and token set

---



## ⚙️ CI/CD with AWS

#### AWS Services Used:

- IAM : Create custom service roles for EC2 and CodeDeploy

- EC2 : Launch instance, install required packages

- S3 : Stores source & build artifacts

- CodeBuild : Builds app from repo using buildspec.yaml

- CodeDeploy : Deploys to EC2 instance

- CodePipeline : Orchestrates CI/CD process

### Plan:

- Create a pipeline that:

- Pulls code from GitHub

- Builds with CodeBuild

- Stores 'SourceArtifact/' & 'BuildArtifact/' in S3 internally managed by CodePipeline

- Deploys to EC2 with CodeDeploy

- Glues entire flow using CodePipeline

#### Before AWS Console Work

- Add 'prod_requirements.txt' into 'flaskapp/'

- Copy-Paste 'logger/' and 'params.yaml' from root dir into 'flaskapp/'

- Add 'scripts/', 'buildspec.yaml', 'appspec.yaml' into root directory

- Git add, commit, and push

#### IAM Configurations:

- Create custom service roles for EC2 and CodeDeploy

- Manage permissions for each service.

#### EC2 Setup:

- Set up as the deployment target.
    
- Attach the appropriate IAM role.
    
- Writing and running a shell script manually using Vim and Bash.

#### S3 Bucket:

- Used by CodePipeline internally to store artifacts.

- Auto-created and managed; remember to delete it after use.

#### CodeBuild Service:
    
- Handles the build phase.
    
- Uses a buildspec to compile/test the app.

#### CodeDeploy Service:
    
- Manages deployment to EC2.
    
- Uses the custom IAM role and an AppSpec file.
---
#### CodePipeline Service:
    
- Orchestrates the entire flow: Source → Build → Deploy (we're skipping test stage).
    
- Connects S3, CodeBuild, and CodeDeploy stages.
---

## Demonstration of Successful Pipeline Execution on CodePipeline Creation
https://github.com/user-attachments/assets/c7038f6a-9025-4e3d-b53f-f9fcef203a95
- Source: Github(via App)
- Build: CodeBuild
- Deploy: CodeDeploy
---

#### 🚤 Deploy and Run

- SSH into EC2

```Terminal
# Navigate to project folder
cd /home/ubuntu/flaskapp

# Installing dependencies required for our flaskapp
pip3 install -r prod_requirements.txt

# Running flaskapp on EC2 instance
python3 app.py
```

#### Authorize with DagsHub if prompted

- Access app in browser:

```Terminal
# Replace Private-IP with Public-IP in given link
http://<'EC2-Public-IP'>:5000
```

#### 📊 Tools & Tech Stack

- Python, Pandas, Scikit-learn

- DVC, MLflow, Flask

- DagsHub, GitHub, AWS S3, EC2, CodeBuild, CodeDeploy, CodePipeline

- Docker (optional), GitHub Actions (optional)


---

## 🙌 Acknowledgements

Inspired by industry-level ML workflows and MLOps best practices.  
Thanks to open-source projects like [DVC](https://dvc.org), [MLflow](https://mlflow.org), and [DagsHub](https://dagshub.com).

---
