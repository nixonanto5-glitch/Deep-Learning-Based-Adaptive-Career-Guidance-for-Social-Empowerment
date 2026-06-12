# Deep-Learning-Based-Adaptive-Career-Guidance-for-Social-Empowerment
An AI-powered career guidance platform that analyzes user skills, interests, and educational background to provide personalized career recommendations.
## Overview

This project is a Flask-based intelligent career guidance system that helps students identify suitable career opportunities based on academic performance and placement-related data. The system uses a hybrid recommendation model and automated placement data collection to provide adaptive career guidance.

## Features

- Student Login System
- Student Dashboard
- Teacher Dashboard
- Hybrid Career Recommendation Model
- Placement Data Analysis
- Automated Placement Data Scraping
- Personalized Career Guidance
- Web-Based Interface

## Project Structure

```
DEEP LEARNING - ADAPTIVE CAREER GUIDANCE/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── 3rdyearstudentsdata.csv
│   └── 4thyearstudentsdata.csv
│
├── models/
│   ├── __init__.py
│   └── hybrid_model.py
│
├── selenium_tools/
│   ├── __init__.py
│   └── placement_scraper.py
│
├── templates/
│   ├── login.html
│   ├── student_dashboard.html
│   └── teacher_dashboard.html
│
└── utils/
    ├── __init__.py
    └── auth_data.py
```

## Technologies Used

- Python
- Flask
- NumPy
- Requests
- BeautifulSoup4
- Selenium
- WebDriver Manager
- HTML
- CSS
- JavaScript

## Requirements

```txt
flask>=3.0.0
numpy>=1.26.0
requests>=2.31.0
beautifulsoup4>=4.12.0
selenium>=4.18.0
webdriver-manager>=4.0.1
```

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

## Dataset

The project uses student academic and placement datasets:

- 3rd Year Student Data
- 4th Year Student Data

These datasets are used to generate adaptive career recommendations and placement insights.

## Modules

### Hybrid Recommendation Model

Generates personalized career guidance using student academic and placement-related information.

### Placement Scraper

Collects placement-related information automatically using Selenium.

### Authentication Module

Handles student and teacher login authentication.

## Future Enhancements

- Deep Learning Model Integration
- Resume Analysis
- Skill Gap Detection
- Job Recommendation Engine
- Career Path Visualization
- Interview Preparation Suggestions

## Author

Nixon Anto

Bachelor of Engineering (Computer Science and Engineering)

## License

This project is developed for academic and educational purposes.
