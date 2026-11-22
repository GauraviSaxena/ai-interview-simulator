"""PDF/DOCX resume text extraction with advanced skill taxonomy, academic subjects, and project detection."""
from __future__ import annotations

import io
import re
from typing import Any

# Comprehensive skill taxonomy spanning Indian IT, Software Engineering, Cloud, AI/ML, and Core Disciplines
SKILLS_TAXONOMY = [
    # Programming Languages
    "Python", "C++", "C", "Java", "TypeScript", "JavaScript", "Go", "Rust", "SQL", "R", "Bash",
    "C#", "PHP", "HTML", "HTML5", "CSS", "CSS3", "SCSS", "Dart", "Kotlin", "Swift", "Scala", "Shell Scripting",
    # Frameworks & Full-Stack Web Development
    "React", "React Native", "Next.js", "Vue", "Vue.js", "Angular", "Node.js", "Express", "Express.js",
    "NestJS", "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "ASP.NET", ".NET", ".NET Core",
    "MERN", "MEAN", "Flutter", "Redux", "Tailwind CSS", "Bootstrap", "REST APIs", "GraphQL", "gRPC", "WebSockets",
    # Databases, Caching & Data Stores
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Oracle", "Cassandra", "DynamoDB",
    "Elasticsearch", "Firebase", "MariaDB", "MS SQL Server", "Neo4j",
    # AI / Machine Learning / Deep Learning / GenAI
    "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn",
    "Computer Vision", "NLP", "Natural Language Processing", "OpenCV", "MediaPipe", "Hugging Face",
    "Transformers", "LLMs", "Large Language Models", "LangChain", "LlamaIndex", "RAG",
    "Fine-Tuning", "Vector Databases", "ChromaDB", "Pinecone", "FAISS", "Prompt Engineering",
    "NLTK", "SpaCy", "Generative AI", "GenAI", "BERT", "GPT",
    # Data Engineering & Analytics
    "Pandas", "NumPy", "SciPy", "Statistics", "A/B Testing", "Experimentation", "Data Analysis",
    "Data Visualization", "Power BI", "Tableau", "Matplotlib", "Seaborn", "Plotly",
    "Kafka", "Spark", "Apache Spark", "Airflow", "Databricks", "Hadoop", "ETL", "Snowflake",
    # Cloud, DevOps, Infrastructure & Tools
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "Git", "GitHub", "GitLab",
    "CI/CD", "GitHub Actions", "Jenkins", "Terraform", "Ansible", "Linux", "Ubuntu", "Nginx", "Apache",
    "Postman", "Swagger", "Jira", "Microservices", "Serverless",
    # Academic Subjects & CS Fundamentals (AKTU / Indian Engineering Placement Focus)
    "OOPs", "Object Oriented Programming", "DBMS", "Database Management Systems",
    "Operating Systems", "Computer Networks", "Data Structures", "Algorithms", "DSA",
    "Software Engineering", "Compiler Design", "Computer Architecture", "System Design",
    "Distributed Systems", "Cloud Computing", "Cybersecurity", "Information Security", "Network Security",
    # Quality Assurance & Testing
    "Unit Testing", "Pytest", "JUnit", "Selenium", "Cypress", "Playwright", "Jest", "Automation Testing", "Manual Testing",
    # Embedded Systems, Core Engineering & Hardware
    "Embedded Systems", "Embedded C", "IoT", "Internet of Things", "Arduino", "Raspberry Pi",
    "VLSI", "Verilog", "VHDL", "Microcontrollers", "MATLAB", "Robotics", "AutoCAD",
    # Emerging Technologies
    "Blockchain", "Ethereum", "Solidity", "Smart Contracts", "Web3",
]

# Academic Subjects Taxonomy for Indian Engineering Placements (AKTU / Delhi-NCR)
ACADEMIC_SUBJECTS_TAXONOMY = {
    "OOPs": ["oops", "object oriented programming", "object-oriented programming", "classes and objects", "polymorphism", "inheritance"],
    "DBMS": ["dbms", "database management", "database management systems", "relational database", "sql", "normalization"],
    "Operating Systems": ["operating systems", "os", "process scheduling", "memory management", "virtual memory", "deadlocks"],
    "Computer Networks": ["computer networks", "networking", "tcp/ip", "osi model", "network protocols"],
    "Data Structures & Algorithms": ["data structures", "algorithms", "dsa", "data structures and algorithms"],
    "Software Engineering": ["software engineering", "sdlc", "agile", "scrum", "software design"],
    "System Design": ["system design", "distributed systems", "microservices architecture", "scalability"],
}

# For backward compatibility
SKILLS = SKILLS_TAXONOMY


def extract_text_from_pdf(data: bytes) -> str:
    """Extract full raw text from PDF binary data."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages_text.append(txt)
    return "\n".join(pages_text)


def extract_text_from_docx(data: bytes) -> str:
    """Extract full raw text from DOCX binary data including tables."""
    from docx import Document
    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs if p.text]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                lines.append(row_text)
    return "\n".join(lines)


def detect_skills(text: str) -> list[str]:
    """Scan resume text against skills taxonomy with word-boundary awareness."""
    lower_text = f" {text.lower()} "
    detected = []
    for skill in SKILLS_TAXONOMY:
        # Match as discrete term or phrase with word boundary
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lower_text):
            detected.append(skill)
    # Deduplicate while preserving order
    return list(dict.fromkeys(detected))


def detect_academic_subjects(text: str) -> list[str]:
    """Detect core computer science subjects from candidate resume text."""
    lower_text = f" {text.lower()} "
    found_subjects = []
    for subject, aliases in ACADEMIC_SUBJECTS_TAXONOMY.items():
        for alias in aliases:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"
            if re.search(pattern, lower_text):
                found_subjects.append(subject)
                break
    return list(dict.fromkeys(found_subjects))


def extract_jd_requirements(jd_text: str) -> dict[str, Any]:
    """Extract core technical requirements, skills, and subjects from a Job Description."""
    clean_jd = (jd_text or "").strip()
    if not clean_jd:
        return {"skills": [], "subjects": [], "role_keywords": []}

    skills = detect_skills(clean_jd)
    subjects = detect_academic_subjects(clean_jd)
    
    # Extract common Indian IT role keywords
    role_signals = [
        "Software Engineer", "SDE", "Full Stack", "Frontend", "Backend",
        "Data Scientist", "Data Analyst", "DevOps Engineer", "Cloud Engineer",
        "QA Engineer", "Automation Tester", "Machine Learning Engineer", "GenAI Engineer"
    ]
    matched_roles = [r for r in role_signals if re.search(r"\b" + re.escape(r) + r"\b", clean_jd, re.IGNORECASE)]

    return {
        "skills": skills,
        "subjects": subjects,
        "role_keywords": matched_roles,
    }


def detect_projects(text: str) -> list[dict[str, Any]]:
    """Heuristically identify project blocks, names, tech stack, and descriptions from resume text."""
    projects: list[dict[str, Any]] = []
    
    # Try finding dedicated project sections
    section_patterns = [
        r"(?:PROJECTS|KEY PROJECTS|NOTABLE PROJECTS|ACADEMIC PROJECTS|PERSONAL PROJECTS|CAPSTONE PROJECTS)[\s\:\-]+(.*?)(?:EXPERIENCE|WORK EXPERIENCE|EDUCATION|SKILLS|CERTIFICATIONS|PUBLICATIONS|ACHIEVEMENTS|$)",
        r"(?:Projects)[\s\:\-]+(.*?)(?:Experience|Education|Skills|Certifications|Achievements|$)",
    ]
    
    section_text = ""
    for pat in section_patterns:
        match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if match:
            section_text = match.group(1).strip()
            break
            
    target_text = section_text if len(section_text) > 40 else text
    
    # Split by project indicators
    lines = [line.strip() for line in target_text.splitlines() if line.strip()]
    current_title = ""
    current_desc: list[str] = []
    
    def _extract_tech(line_or_desc: str) -> list[str]:
        return detect_skills(line_or_desc)

    for line in lines:
        is_title = False
        # Title heuristic: short line without ending punctuation or containing keywords like '|', '•', '-', 'using'
        if len(line) < 95 and not line.endswith((".", ";")):
            if any(marker in line for marker in ["|", "•", " - ", "Tech Stack:", "Technologies:", "Tools:"]):
                is_title = True
            elif re.match(r"^(\d+\.|\*|\-)?\s*[A-Z0-9][A-Za-z0-9\s\-]{3,50}(?:\:|\s*\(|$)", line):
                is_title = True
                
        if is_title and current_desc:
            desc_combined = " ".join(current_desc[:4])
            proj_skills = _extract_tech(f"{current_title} {desc_combined}")
            projects.append({
                "title": current_title or "Key Project",
                "description": desc_combined,
                "tech_stack": proj_skills[:5],
            })
            current_title = line
            current_desc = []
        elif is_title and not current_title:
            current_title = line
        else:
            current_desc.append(line)
            
    if current_title and current_desc:
        desc_combined = " ".join(current_desc[:4])
        proj_skills = _extract_tech(f"{current_title} {desc_combined}")
        projects.append({
            "title": current_title,
            "description": desc_combined,
            "tech_stack": proj_skills[:5],
        })
        
    # If heuristic yielded no projects, create structured summaries from action sentences
    if not projects:
        action_sentences = re.findall(
            r"(?:Developed|Built|Designed|Implemented|Trained|Architected|Created|Deployed)[^\.\n]+(?:\.|\n)",
            text,
            re.IGNORECASE,
        )
        for i, sentence in enumerate(action_sentences[:3]):
            proj_skills = _extract_tech(sentence)
            projects.append({
                "title": f"Flagship Technical Project #{i+1}",
                "description": sentence.strip(),
                "tech_stack": proj_skills[:5],
            })
            
    return projects[:4]


def extract_resume(file_obj: Any, filename: str | None = None) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Extract text from an uploaded PDF or DOCX file, detect skills, and extract project evidence.
    
    Returns:
        tuple of (clean_text: str, skills: list[str], projects: list[dict[str, Any]])
    """
    if file_obj is None:
        return "", [], []

    name = (filename or getattr(file_obj, "name", "") or "").lower()

    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
    elif hasattr(file_obj, "read"):
        data = file_obj.read()
    elif isinstance(file_obj, (bytes, bytearray, memoryview)):
        data = bytes(file_obj)
    else:
        raise TypeError("Resume input must be an uploaded file object or raw bytes.")

    if not name and isinstance(file_obj, (bytes, bytearray, memoryview)):
        raise ValueError("A filename ending in .pdf or .docx is required when passing raw bytes.")

    if not isinstance(data, bytes):
        data = bytes(data)

    text = ""
    try:
        if name.endswith(".pdf"):
            text = extract_text_from_pdf(data)
        elif name.endswith(".docx"):
            text = extract_text_from_docx(data)
        else:
            raise ValueError("Please upload a PDF or DOCX resume.")
    except ImportError as exc:
        dep = "pypdf" if name.endswith(".pdf") else "python-docx"
        raise RuntimeError(f"Missing dependency {dep} to parse resume. Paste resume text as fallback.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError(f"Resume parsing error: {exc}. You can paste resume text instead.") from exc

    clean_text = re.sub(r"\s+", " ", text).strip()
    skills = detect_skills(clean_text)
    projects = detect_projects(clean_text)

    return clean_text, skills, projects


def extract_text_from_file(file_obj: Any, filename: str = "") -> str:
    """Extract clean text from an uploaded file object or raw bytes."""
    clean_text, _, _ = extract_resume(file_obj, filename)
    return clean_text


def extract_skills_and_projects(text: str) -> dict[str, Any]:
    """Detect skills, academic subjects, and projects from raw resume text."""
    skills = detect_skills(text)
    projects = detect_projects(text)
    subjects = detect_academic_subjects(text)
    return {
        "skills": skills,
        "projects": projects,
        "academic_subjects": subjects,
    }


def validate_target_role(role: str) -> tuple[bool, str]:
    from modules.llm_evaluator import validate_target_role as _vtr
    return _vtr(role)
