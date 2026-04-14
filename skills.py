# skills.py
# ─────────────────────────────────────────────────────────────────────────────
# Skills Extraction Module
# Maintains a curated dictionary of tech/domain skills and extracts them from
# resume text using exact and fuzzy matching.
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import List

# ── Master Skills Dictionary ──────────────────────────────────────────────────
SKILLS_DB = {
    # Programming Languages
    "programming_languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
        "Go", "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R",
        "MATLAB", "Perl", "Shell", "Bash", "PowerShell", "Dart", "Lua",
        "Haskell", "Elixir", "Clojure", "F#", "Julia",
    ],

    # Web Frameworks & Libraries
    "web_frameworks": [
        "React", "Angular", "Vue.js", "Next.js", "Nuxt.js", "Svelte",
        "Django", "Flask", "FastAPI", "Spring Boot", "Express", "Node.js",
        "Laravel", "Ruby on Rails", "ASP.NET", "GraphQL", "REST API",
        "WebSocket", "Tailwind CSS", "Bootstrap", "Sass", "jQuery",
    ],

    # Data Science & ML
    "data_science_ml": [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost",
        "LightGBM", "CatBoost", "Pandas", "NumPy", "SciPy", "Matplotlib",
        "Seaborn", "Plotly", "OpenCV", "BERT", "GPT", "Transformers",
        "Hugging Face", "YOLO", "Random Forest", "Logistic Regression",
        "Neural Network", "Reinforcement Learning", "A/B Testing",
    ],

    # Databases
    "databases": [
        "MySQL", "PostgreSQL", "SQLite", "Oracle", "SQL Server", "MongoDB",
        "Redis", "Cassandra", "DynamoDB", "Elasticsearch", "Neo4j",
        "CouchDB", "Firebase", "Supabase", "Snowflake", "BigQuery",
        "Redshift", "HBase",
    ],

    # Cloud Platforms
    "cloud": [
        "AWS", "Azure", "Google Cloud", "GCP", "Heroku", "DigitalOcean",
        "Cloudflare", "Vercel", "Netlify", "EC2", "S3", "Lambda",
        "RDS", "CloudFormation", "Terraform", "Pulumi",
    ],

    # DevOps & Tools
    "devops": [
        "Docker", "Kubernetes", "Jenkins", "GitHub Actions", "GitLab CI",
        "CircleCI", "Travis CI", "Ansible", "Puppet", "Chef", "Helm",
        "ArgoCD", "Prometheus", "Grafana", "ELK Stack", "Splunk",
        "Nginx", "Apache", "Linux", "Unix", "Git", "GitHub", "GitLab",
        "Bitbucket", "Jira", "Confluence", "CI/CD",
    ],

    # Mobile Development
    "mobile": [
        "Android", "iOS", "Flutter", "React Native", "Xamarin",
        "Swift", "SwiftUI", "Kotlin", "Jetpack Compose", "Ionic",
        "Cordova", "Unity",
    ],

    # Data Engineering
    "data_engineering": [
        "Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Flink",
        "Hive", "Pig", "Sqoop", "Informatica", "Talend", "SSIS",
        "ETL", "Data Warehouse", "Data Pipeline",
    ],

    # Cybersecurity
    "cybersecurity": [
        "Penetration Testing", "Ethical Hacking", "SIEM", "Wireshark",
        "Metasploit", "Burp Suite", "OWASP", "Nmap", "Kali Linux",
        "Firewall", "IDS", "IPS", "Zero Trust", "IAM", "GDPR",
        "ISO 27001", "SOC 2", "CEH", "CISSP",
    ],

    # Networking
    "networking": [
        "TCP/IP", "DNS", "DHCP", "VPN", "LAN", "WAN", "BGP", "OSPF",
        "MPLS", "SD-WAN", "Cisco", "Juniper", "Wireshark", "VoIP",
        "4G", "5G", "RF", "Network Security",
    ],

    # Soft Skills / Methodologies
    "methodologies": [
        "Agile", "Scrum", "Kanban", "DevOps", "TDD", "BDD", "Microservices",
        "SOA", "REST", "SOAP", "Design Patterns", "System Design",
        "Object-Oriented", "Functional Programming",
    ],

    # BI & Analytics
    "bi_analytics": [
        "Tableau", "Power BI", "Looker", "Excel", "Google Analytics",
        "Mixpanel", "SQL", "SSRS", "Crystal Reports",
    ],
}

# Flatten the skills list for fast lookup (lowercase → original)
_SKILL_LOOKUP: dict[str, str] = {}
for category_skills in SKILLS_DB.values():
    for skill in category_skills:
        _SKILL_LOOKUP[skill.lower()] = skill


def extract_skills(text: str) -> List[str]:
    """
    Extract known skills from resume text.

    Strategy:
      1. Normalise text.
      2. For each skill in our DB, check if it appears as a whole word/phrase.
      3. De-duplicate while preserving order.

    Args:
        text: Raw or lightly cleaned resume text.

    Returns:
        Sorted list of matched skill strings.
    """
    if not text:
        return []

    text_lower = text.lower()
    found: dict[str, str] = {}  # lower_skill → original_skill

    for skill_lower, skill_original in _SKILL_LOOKUP.items():
        # Build a regex that matches the skill as a whole word (handles "C++" etc.)
        escaped = re.escape(skill_lower)
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found[skill_lower] = skill_original

    return sorted(found.values(), key=lambda s: s.lower())


def categorise_skills(skills: List[str]) -> dict:
    """
    Group extracted skills by category.

    Args:
        skills: List of skill strings (as returned by extract_skills).

    Returns:
        Dict mapping category name → list of skills.
    """
    result: dict[str, List[str]] = {}
    skill_set = {s.lower() for s in skills}

    for category, category_skills in SKILLS_DB.items():
        matched = [s for s in category_skills if s.lower() in skill_set]
        if matched:
            result[category] = matched

    return result


def get_skill_count(text: str) -> int:
    """Return the number of distinct skills found in text."""
    return len(extract_skills(text))


if __name__ == "__main__":
    sample = (
        "Experienced data scientist with Python, TensorFlow, PyTorch, and scikit-learn. "
        "Worked with AWS, Docker, and Kubernetes. Strong SQL and MongoDB skills. "
        "Experience in NLP using BERT and Hugging Face Transformers."
    )
    skills = extract_skills(sample)
    print("Extracted Skills:", skills)
    print("\nCategorised:")
    for cat, sk in categorise_skills(skills).items():
        print(f"  {cat}: {sk}")
# Skill Gap Mapping for Job Roles
ROLE_REQUIRED_SKILLS = {
    "Data Scientist": ["Python", "Machine Learning", "Pandas", "NumPy", "SQL", "Statistics"],
    "AI Engineer": ["Python", "Deep Learning", "TensorFlow", "PyTorch", "NLP"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Data Analyst": ["Excel", "SQL", "Python", "Power BI"],
    "Software Engineer": ["Java", "Python", "Data Structures", "Algorithms", "Git"]
}

def skill_gap(user_skills, role):
    required = ROLE_REQUIRED_SKILLS.get(role, [])
    missing = list(set(required) - set(user_skills))
    return missing