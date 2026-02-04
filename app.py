import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Resume Screener - GenAI Delivery Lead",
    page_icon="📋",
    layout="wide"
)

# Model configuration
MODEL = "gpt-5.2"

# Prompt template
SCREENING_PROMPT = """# Task

Review the provided resume against the Guidance provided and basis that recommend if we should proceed with the first round of interview or not

# Guidance

The core problem to solve
We are not hiring for an LLM engineer or ML researcher.
We are hiring for a GenAI Product/Program Lead (Life Sciences) who can translate pharma use-cases into build-ready requirements, run delivery with engineering, and drive quality + adoption in regulated workflows.
Think: "AI Transformation PM / AI Product Ops / GenAI Delivery Lead" — not "Software Engineer" or "Data Scientist".
________________________________________
Role archetype
•    Senior Associate – GenAI Productization & Delivery (Life Sciences)
•    GenAI Implementation Lead – Life Sciences Platforms
•    AI Transformation Program Manager – Pharma
Avoid: "GenAI Engineer", "ML Engineer", "Data Scientist" (these attract the wrong pipeline).
________________________________________
What success looks like:
This person should be able to:
1.    Take a client problem statement → convert it into workflow + prompt/eval requirements
2.    Coordinate with Engineering to ship it → without writing core code themselves
3.    Define "good" → quality rubric, evaluation plan, traceability, feedback loop
4.    Drive adoption → training, governance, metrics, iterations
________________________________________
Must-have skills (non-negotiable)
1) GenAI implementation literacy (not research)
•    Prompting patterns, RAG basics, citations/traceability
•    Eval concepts (LLM-as-judge basics, rubrics, test cases, regression mindset)
•    Ability to reason about hallucinations, grounding, failure modes
2) Program/product execution
•    Can run cross-functional delivery with Eng + SMEs + stakeholders
•    Writes crisp PRDs / user stories / acceptance criteria
•    Strong on dependency management, risks, timelines, stakeholder updates
3) Regulated-domain comfort (life sciences preferred)
•    Has worked in pharma / healthcare / medtech OR adjacent regulated enterprise workflows
•    Understands why "accuracy, traceability, and reviewability" matter
4) Communication and structuring
•    Can synthesize messy inputs into structured specs and decisions
•    Can demo/communicate to business stakeholders confidently
________________________________________
Good-to-have skills (strong signals)
•    Experience shipping workflow tools (authoring, review, compliance, PV, med info, regulatory)
•    Exposure to document-heavy systems (PDF/Word extraction, templates, knowledge bases)
•    Basic SQL / analytics for adoption metrics
•    UX collaboration experience (wireframes, feedback loops)
________________________________________
What we do not need
•    Not looking for LLM researchers, model trainers, or deep ML (PyTorch, Transformers training).
•    Not looking for pure backend/frontend engineers as the primary fit.
•    Not looking for Kaggle/academic ML profiles without enterprise delivery experience.
(We already have engineering; we need the "glue" leader who makes GenAI real in production.)
________________________________________
Ideal background (what TAG can screen for)
Education
•    MBA + STEM, B.Pharm/Bio/Healthcare + MBA, Engineering + product/program experience — many combinations work.
•    Degrees are less important than proof of shipping + stakeholder leadership.
Experience
•    4–8 years overall (for someone reporting to you; adjust as needed)
•    Worked as one of:
o    Product Manager (enterprise / workflow products)
o    Program Manager (platform delivery)
o    Solutions Consultant / Implementation Lead
o    Digital transformation lead (healthcare/pharma)
o    Analytics-to-product transition profiles who shipped tools
Industries to source from:
•    Life sciences services (Indegene-like)
•    Health-tech (B2B)
•    Enterprise SaaS implementation (especially regulated clients)
•    Consulting (healthcare/tech transformation)
________________________________________
Sample JD snippet (TAG-ready, copy-paste)
Role: GenAI Productization & Delivery Lead (Life Sciences)
We're hiring someone to lead the translation of life sciences GenAI use-cases into production-ready workflows. This role partners with Engineering and domain SMEs to define requirements, run delivery, establish quality/evaluation standards, and drive adoption for GenAI capabilities in regulated, document-heavy environments.
Must have: GenAI implementation literacy (prompting/RAG/evals), strong program/product execution, stakeholder management, and experience in healthcare/pharma or another regulated enterprise domain.
Not required: Model training, deep ML research, advanced coding.
________________________________________
Screening keywords for TAG (send this list)
Target keywords:
GenAI implementation, AI transformation, productization, prompt engineering (applied), RAG, evaluation, LLM QA, workflow automation, enterprise SaaS, program management, product ops, solutions/implementation, regulated domain, life sciences, medtech, pharmacovigilance, medical writing, compliance, traceability.
Reject / deprioritize keywords (unless paired with delivery experience):
Pytorch, model training, fine-tuning LLMs, research publications, Kaggle grandmaster, computer vision research, "built transformer from scratch".
________________________________________
Quick scorecard TAG can use (simple)
•    GenAI literacy (Applied) – 0/1
•    Enterprise delivery (PRD / execution) – 0/1
•    Stakeholder mgmt + communication – 0/1
•    Regulated / healthcare familiarity – 0/1

Hire pipeline: only shortlist candidates with 3/4+.

# Resume

{resume}

# Output Format

Provide your analysis in the following format:

## Scorecard
| Criteria | Score | Evidence |
|----------|-------|----------|
| GenAI literacy (Applied) | 0 or 1 | Brief evidence from resume |
| Enterprise delivery (PRD / execution) | 0 or 1 | Brief evidence from resume |
| Stakeholder mgmt + communication | 0 or 1 | Brief evidence from resume |
| Regulated / healthcare familiarity | 0 or 1 | Brief evidence from resume |

**Total Score: X/4**

## Verdict
**PROCEED TO INTERVIEW** or **DO NOT PROCEED**

## Key Strengths
- Bullet points of relevant strengths

## Concerns / Gaps
- Bullet points of concerns or missing qualifications

## Summary
2-3 sentence summary of your recommendation.
"""


def get_api_key():
    """Get OpenAI API key from secrets or session state."""
    api_key = None

    # Try to get from Streamlit secrets first (for deployment)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Fall back to session state (for manual entry)
    if not api_key and "openai_api_key" in st.session_state:
        api_key = st.session_state.openai_api_key

    return api_key


def analyze_resume(resume_text: str, api_key: str) -> str:
    """Send resume to GPT-5.2 for analysis."""
    client = OpenAI(api_key=api_key)

    prompt = SCREENING_PROMPT.format(resume=resume_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# Main UI
st.title("Resume Screener")
st.subheader("GenAI Productization & Delivery Lead (Life Sciences)")

st.markdown("---")

# API Key handling
api_key = get_api_key()

if not api_key:
    st.warning("OpenAI API key not configured. Please enter your API key below.")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input
        st.rerun()
else:
    st.success("API configured")

st.markdown("---")

# Resume input
st.markdown("### Paste Resume Content Below")
resume_input = st.text_area(
    "Resume",
    height=400,
    placeholder="Paste the candidate's resume content here...",
    label_visibility="collapsed"
)

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

# Process
if analyze_btn:
    api_key = get_api_key()
    if not resume_input.strip():
        st.error("Please paste resume content before analyzing.")
    elif not api_key:
        st.error("Please configure your OpenAI API key first.")
    else:
        with st.spinner("Analyzing resume against role criteria..."):
            try:
                result = analyze_resume(resume_input, api_key)
                st.markdown("---")
                st.markdown("## Analysis Result")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error analyzing resume: {str(e)}")

# Footer
st.markdown("---")
st.caption("This tool uses OpenAI GPT-5.2 to evaluate resumes against predefined criteria for the GenAI Delivery Lead role.")
