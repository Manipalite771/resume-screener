import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import base64
import requests
import tempfile
import os
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Resume Screener - GenAI Delivery Lead",
    page_icon="📋",
    layout="wide"
)

# Model configuration
OPENAI_MODEL = "gpt-5.2"
GEMINI_MODEL = "gemini-3-pro-preview"
GEMINI_API_KEY = "AIzaSyB-Z4ucSSHHUynsx5Ecg1qu6k96-XUbSIQ"

# Resume Quality Review Prompt (Gemini)
QUALITY_REVIEW_PROMPT = """You are a professional resume quality reviewer. Analyze the provided resume image(s) and evaluate the document quality.

## IMPORTANT CONTEXT
- Current date: February 2026. Dates up to 2026 are valid and acceptable.
- First, check if this is an AGENCY CV (look for agency name/logo at the top like "ABC Staffing", "XYZ Recruiters", etc.) or a DIRECT CANDIDATE CV.

## CV Source Detection
- **Agency CV**: Has agency branding/name at top, often missing candidate contact details (this is intentional - agencies remove contact info to prevent direct outreach)
- **Direct CV**: No agency branding, should have candidate's own contact information

## Evaluation Criteria

### 1. **Spelling & Grammar** (0 or 1)
- Check for spelling mistakes, typos, grammatical errors
- Minor capitalization inconsistencies (e.g., "Zero-shot" vs "zero-shot") should NOT fail this criterion
- Focus on actual language errors, not stylistic choices

### 2. **Factual/Technical Consistency** (0 or 1)
- Check if dates are logical (no overlapping employment, dates should be <= 2026)
- Unexplained employment gaps > 1 year should be noted
- **IMPORTANT**: Check for technical inaccuracies in claims, e.g.:
  - Claiming PostgreSQL as a vector database (it's not, unless using pgvector extension)
  - Misrepresenting technologies or their capabilities
  - Inconsistent tech stack claims (e.g., using a framework that didn't exist at the claimed time)
- Verify education timeline makes sense relative to experience

### 3. **Layout & Structure** (0 or 1)
- Is the resume well-organized with clear sections?
- Professional appearance, not cluttered or chaotic
- **DO NOT penalize** for:
  - Minor bullet point style variations (solid vs hollow)
  - Slight spacing inconsistencies
  - These may be artifacts from agency reformatting, not candidate's fault

### 4. **Attention to Detail** (0 or 1)
- **For Agency CVs**:
  - Missing contact info is EXPECTED and should NOT be penalized
  - Focus on content quality, not formatting issues that agency may have introduced
- **For Direct Candidate CVs**:
  - Missing contact information IS a valid concern
  - Consistent date formats expected
- For both: Focus on quality of work descriptions, clarity of achievements, and professional presentation of experience

## Output Format

Provide your analysis in EXACTLY this JSON format (no other text):
```json
{
  "cv_source": "AGENCY" or "DIRECT",
  "agency_name": "Name if agency CV, otherwise null",
  "spelling_grammar": {"score": 0 or 1, "issues": ["list of significant issues only"]},
  "factual_consistency": {"score": 0 or 1, "issues": ["list of factual/technical errors"]},
  "layout_structure": {"score": 0 or 1, "issues": ["list of major structural issues only"]},
  "attention_to_detail": {"score": 0 or 1, "issues": ["list of significant issues based on CV source"]},
  "total_score": X,
  "verdict": "PASS" or "FAIL",
  "summary": "Brief 1-2 sentence summary"
}
```

**SCORING RULES**:
- Total score = sum of all four criteria (max 4)
- PASS if total_score >= 3, otherwise FAIL
- Be lenient on formatting/style issues - focus on SUBSTANCE
- Technical inaccuracies and factual errors are more important than formatting
- Agency CVs should be judged primarily on content quality, not presentation
"""

# Role Screening Prompt (GPT-5.2)
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

# Quality Review Result

{quality_review}

# Resume

{resume}

# Output Format

Provide your analysis in the following format:

## Resume Quality Review
{quality_summary}

## Role Fit Scorecard
| Criteria | Score | Evidence |
|----------|-------|----------|
| GenAI literacy (Applied) | 0 or 1 | Brief evidence from resume |
| Enterprise delivery (PRD / execution) | 0 or 1 | Brief evidence from resume |
| Stakeholder mgmt + communication | 0 or 1 | Brief evidence from resume |
| Regulated / healthcare familiarity | 0 or 1 | Brief evidence from resume |

**Role Fit Score: X/4**
**Quality Penalty: {penalty}**
**Final Score: X/4**

## Verdict
**PROCEED TO INTERVIEW** or **DO NOT PROCEED**

(Note: Candidates need final score of 3/4+ to proceed. Quality review FAIL results in -1 penalty.)

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


def convert_pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of images with base64 encoding."""
    images_data = []

    # Open PDF from bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num, page in enumerate(doc):
        # Render page to image at 150 DPI for good quality
        pix = page.get_pixmap(dpi=150)

        # Convert to PIL Image
        img_data = pix.tobytes("png")

        # Encode to base64
        base64_data = base64.b64encode(img_data).decode('utf-8')

        images_data.append({
            'data': base64_data,
            'mime_type': 'image/png',
            'page_num': page_num + 1
        })

    doc.close()
    return images_data


def call_gemini_with_images(images_data, prompt):
    """Call Gemini API with images for text extraction or quality review."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    parts = [{"text": prompt}]

    for img in images_data:
        parts.append({
            "inline_data": {
                "mime_type": img['mime_type'],
                "data": img['data']
            }
        })

    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 32000
        }
    }

    headers = {"Content-Type": "application/json"}

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, verify=False, timeout=180)

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts_result = candidate['content']['parts']
                        if len(parts_result) > 0 and 'text' in parts_result[0]:
                            return parts_result[0]['text']
                return None
            elif response.status_code == 429:
                import time
                time.sleep(retry_delay * 2)
                retry_delay *= 2
            else:
                st.error(f"API Error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            st.error(f"API Exception: {str(e)}")

        if attempt < max_retries - 1:
            import time
            time.sleep(retry_delay)
            retry_delay *= 2

    return None


def extract_resume_text(images_data):
    """Extract text from resume images using Gemini."""
    extraction_prompt = """Extract ALL text content from this resume image(s).

IMPORTANT:
- Extract text EXACTLY as written - preserve all details
- Maintain the structure (sections, bullet points, etc.)
- Include all dates, company names, job titles, skills, education details
- Do not summarize or paraphrase - extract verbatim
- If there are multiple pages, process them in order

Output the complete resume text in a clean, readable format."""

    return call_gemini_with_images(images_data, extraction_prompt)


def perform_quality_review(images_data):
    """Perform quality review on resume images using Gemini."""
    return call_gemini_with_images(images_data, QUALITY_REVIEW_PROMPT)


def parse_quality_review(quality_response):
    """Parse the quality review JSON response."""
    import json
    import re

    try:
        # Extract JSON from response (it might be wrapped in markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', quality_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_str = quality_response

        # Clean up the string
        json_str = json_str.strip()

        # Parse JSON
        quality_data = json.loads(json_str)
        return quality_data
    except Exception as e:
        st.warning(f"Could not parse quality review response: {e}")
        # Return default PASS if parsing fails
        return {
            "cv_source": "UNKNOWN",
            "agency_name": None,
            "total_score": 4,
            "verdict": "PASS",
            "summary": "Quality review parsing failed - defaulting to PASS",
            "spelling_grammar": {"score": 1, "issues": []},
            "factual_consistency": {"score": 1, "issues": []},
            "layout_structure": {"score": 1, "issues": []},
            "attention_to_detail": {"score": 1, "issues": []}
        }


def analyze_resume(resume_text: str, quality_data: dict, api_key: str) -> str:
    """Send resume to GPT-5.2 for analysis with quality review context."""
    client = OpenAI(api_key=api_key)

    # Determine penalty
    quality_verdict = quality_data.get('verdict', 'PASS')
    penalty = "-1" if quality_verdict == "FAIL" else "0"

    # Get CV source info
    cv_source = quality_data.get('cv_source', 'UNKNOWN')
    agency_name = quality_data.get('agency_name')
    cv_source_text = f"Agency CV ({agency_name})" if cv_source == "AGENCY" and agency_name else cv_source

    # Build quality summary
    quality_summary = f"""
**CV Source: {cv_source_text}**
**Quality Verdict: {quality_verdict}**
- Spelling & Grammar: {quality_data.get('spelling_grammar', {}).get('score', 'N/A')}/1
- Factual/Technical Consistency: {quality_data.get('factual_consistency', {}).get('score', 'N/A')}/1
- Layout & Structure: {quality_data.get('layout_structure', {}).get('score', 'N/A')}/1
- Attention to Detail: {quality_data.get('attention_to_detail', {}).get('score', 'N/A')}/1
- Quality Score: {quality_data.get('total_score', 'N/A')}/4
- Summary: {quality_data.get('summary', 'N/A')}
"""

    # Build quality review context for GPT
    quality_review_context = f"""
The resume has undergone a quality review with the following results:
- CV Source: {cv_source_text}
- Verdict: {quality_verdict}
- Quality Score: {quality_data.get('total_score', 'N/A')}/4
- Issues Found: {quality_data.get('summary', 'None noted')}

{'IMPORTANT: Since quality review FAILED, apply a -1 penalty to the final score.' if quality_verdict == 'FAIL' else 'Quality review passed - no penalty applied.'}
"""

    prompt = SCREENING_PROMPT.format(
        resume=resume_text,
        quality_review=quality_review_context,
        quality_summary=quality_summary,
        penalty=penalty
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
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
    st.warning("Please enter your access key to continue.")
    api_key_input = st.text_input(
        "Access Key",
        type="password",
        help="Enter your access key"
    )
    if api_key_input:
        st.session_state.openai_api_key = api_key_input
        st.rerun()
else:
    st.success("Ready to use")

st.markdown("---")

# Resume upload
st.markdown("### Upload Resume (PDF only)")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload the candidate's resume in PDF format"
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

# Process
if analyze_btn:
    api_key = get_api_key()
    if not uploaded_file:
        st.error("Please upload a PDF resume before analyzing.")
    elif not api_key:
        st.error("Please enter your access key first.")
    else:
        # Read PDF bytes
        pdf_bytes = uploaded_file.read()

        # Step 1: Convert PDF to images
        with st.spinner("Converting PDF to images..."):
            try:
                images_data = convert_pdf_to_images(pdf_bytes)
                st.info(f"Processed {len(images_data)} page(s)")
            except Exception as e:
                st.error(f"Error converting PDF: {str(e)}")
                st.stop()

        # Step 2: Quality Review with Gemini
        with st.spinner("Performing resume quality review..."):
            try:
                quality_response = perform_quality_review(images_data)
                if quality_response:
                    quality_data = parse_quality_review(quality_response)
                else:
                    st.warning("Quality review failed - proceeding with default PASS")
                    quality_data = {"verdict": "PASS", "total_score": 4, "summary": "Review unavailable"}
            except Exception as e:
                st.warning(f"Quality review error: {str(e)} - proceeding with default PASS")
                quality_data = {"verdict": "PASS", "total_score": 4, "summary": "Review unavailable"}

        # Display quality review result
        st.markdown("---")
        st.markdown("### Step 1: Resume Quality Review")

        # Show CV source
        cv_source = quality_data.get('cv_source', 'UNKNOWN')
        agency_name = quality_data.get('agency_name')

        if cv_source == "AGENCY":
            source_text = f"Agency CV"
            if agency_name:
                source_text += f" ({agency_name})"
            st.info(f"📋 **CV Source:** {source_text} - Formatting leniency applied")
        else:
            st.info(f"📋 **CV Source:** Direct Candidate CV")

        quality_verdict = quality_data.get('verdict', 'PASS')
        if quality_verdict == "PASS":
            st.success(f"Quality Review: **PASS** ({quality_data.get('total_score', 'N/A')}/4)")
        else:
            st.error(f"Quality Review: **FAIL** ({quality_data.get('total_score', 'N/A')}/4) - This will result in a -1 penalty to the final score")

        with st.expander("View Quality Review Details"):
            st.markdown(f"**Summary:** {quality_data.get('summary', 'N/A')}")

            for criterion in ['spelling_grammar', 'factual_consistency', 'layout_structure', 'attention_to_detail']:
                criterion_data = quality_data.get(criterion, {})
                score = criterion_data.get('score', 'N/A')
                issues = criterion_data.get('issues', [])

                criterion_name = criterion.replace('_', ' ').title()
                score_icon = "✅" if score == 1 else "❌"

                st.markdown(f"**{criterion_name}:** {score_icon} ({score}/1)")
                if issues and len(issues) > 0:
                    for issue in issues:
                        st.markdown(f"  - {issue}")

        # Clear indicator that this is not the final verdict
        st.warning("⚠️ **This is NOT the final verdict.** Quality review only affects scoring. The final PROCEED/DO NOT PROCEED decision is based on Role Fit Analysis below.")

        # Step 3: Extract text from resume
        with st.spinner("Extracting resume content..."):
            try:
                resume_text = extract_resume_text(images_data)
                if not resume_text:
                    st.error("Failed to extract text from resume")
                    st.stop()
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                st.stop()

        # Step 4: Analyze with GPT-5.2
        with st.spinner("Analyzing resume against role criteria..."):
            try:
                result = analyze_resume(resume_text, quality_data, api_key)

                # Extract verdict from result
                import re
                verdict_match = re.search(r'\*\*(PROCEED TO INTERVIEW|DO NOT PROCEED)\*\*', result)
                final_verdict = verdict_match.group(1) if verdict_match else None

                # Extract final score if present
                score_match = re.search(r'\*\*Final Score:\s*(\d+)/4\*\*', result)
                final_score = score_match.group(1) if score_match else None

                st.markdown("---")

                # Prominent verdict display
                st.markdown("## Final Decision")

                if final_verdict == "PROCEED TO INTERVIEW":
                    st.success(f"""
                    ## ✅ PROCEED TO INTERVIEW
                    **Final Score: {final_score}/4** (minimum 3/4 required)
                    """)
                elif final_verdict == "DO NOT PROCEED":
                    st.error(f"""
                    ## ❌ DO NOT PROCEED
                    **Final Score: {final_score}/4** (minimum 3/4 required)
                    """)
                else:
                    st.warning("⚠️ Could not determine verdict - please review analysis below")

                # Detailed analysis in expander
                with st.expander("📋 View Detailed Analysis", expanded=False):
                    st.markdown(result)

            except Exception as e:
                st.error(f"Error analyzing resume: {str(e)}")

# Footer
st.markdown("---")
st.caption("Resume screening powered by AI")
