import ollama

def llama_generate(prompt):
    try:
        response = ollama.chat(
            model='tinyllama',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error running TinyLLaMA: {e}"


def generate_resume_summary(raw_text, skills, role, cname):
    prompt = f"""
    Write a professional resume summary for {cname} applying for {role}.
    Highlight relevant skills: {', '.join(skills)}.
    """
    return llama_generate(prompt)


def generate_cover_letter(raw_text, skills, role, company_name=None, job_description=None, candidate_name=None):
    prompt = f"""
    Write a professional cover letter for {candidate_name}
    applying for {role} at {company_name}.
    Job description: {job_description}
    Skills: {', '.join(skills)}
    """
    return llama_generate(prompt)


def generate_interview_tips(skills, role):
    prompt = f"""
    Generate interview questions and preparation tips for {role}.
    Skills: {', '.join(skills)}
    """
    return llama_generate(prompt)


def generate_career_roadmap(role, skills):
    prompt = f"""
    Create a career roadmap to become a {role}.
    Current skills: {', '.join(skills)}.
    Include technologies to learn, projects to build, and career steps.
    """
    return llama_generate(prompt)