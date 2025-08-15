import streamlit as st
import os, json, time, configparser, re
from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

# Optional deps
from db_client import get_db, mongo_ping

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Conversational Banking – Pre‑POC (v4)", layout="wide")

def load_cfg():
    cfg = configparser.ConfigParser()
    try:
        files = cfg.read("config.ini", encoding="utf-8")
        if not files:
            st.error("[ERROR] config.ini not found or unreadable.")
    except Exception as e:
        st.error(f"[ERROR] Failed to load config.ini: {e}")
        st.write(f"[DEBUG] Exception loading config.ini: {e}")
    return cfg

## get_mongo is now replaced by get_db from db.client

@st.cache_data
def get_questions(_cfg: configparser.ConfigParser) -> List[Dict[str, Any]]:
    data = _cfg["QUESTIONS"]["questions_json"]
    return json.loads(data)

def openai_followups(k: int, sys_prompt: str, user_tmpl: str, answer: str, model: str, temperature: float, max_tokens: int) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        # deterministic fallback
        return [
            "Which systems/APIs are involved here?",
            "How will you measure success in this area?",
            "What security or compliance constraints apply?",
            "Who owns this process end-to-end?",
            "What is the main failure mode today?"
        ][:k]
    client = OpenAI(api_key=api_key)
    try:
        content = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":user_tmpl.format(answer=answer)}
            ],
            max_tokens=max_tokens
        ).choices[0].message.content.strip()
        # Remove code block markers and filter out junk
        content = re.sub(r"^```[a-zA-Z]*", "", content)
        content = content.replace("```", "").strip()
        # Try JSON parse first
        try:
            arr = json.loads(content)
            if isinstance(arr, list):
                arr = [str(x).strip() for x in arr if x and isinstance(x, str) and len(x.strip()) > 5]
                if arr:
                    return arr[:k]
        except Exception:
            pass
        # Fallback: split lines, filter out short/junk lines
        lines = [ln.strip("- •* ").strip() for ln in content.splitlines() if ln.strip()]
        # Remove lines that are just '[', ']', 'ok', or too short
        clean_lines = [ln for ln in lines if ln not in ("[", "]", "ok", "", "null") and len(ln) > 5]
        if clean_lines:
            return clean_lines[:k]
        # Final fallback
        return ["Please provide more details.","Any metrics?","Any blockers?","Owners?","Risks?"][:k]
    except Exception as e:
        st.warning(f"Follow-up generation failed; using defaults. ({e})")
        return ["Please provide more details.","Any metrics?","Any blockers?","Owners?","Risks?"][:k]

def login_screen(cfg):
    st.title(cfg["APP"]["name"] + " (v4)")
    st.subheader("Sign in")
    role = st.selectbox("Role", ["User","Head","Admin","Data Infrastructure"], index=0, help="Choose your role to continue.")
    username = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    ok = st.button("Login", type="primary")
    if ok:
        auth = cfg["AUTH"]
        valid = (
            (role=="User" and pwd==auth.get("user_password","")) or
            (role=="Head" and pwd==auth.get("head_password","")) or
            (role=="Admin" and pwd==auth.get("admin_password","")) or
            (role=="Data Infrastructure" and pwd==auth.get("data_infrastructure_password",""))
        )
        st.write(f"[DEBUG] Login attempted for role: {role}, username: {username}, valid: {valid}")
        if valid and username.strip():
            st.session_state["role"] = role
            st.session_state["username"] = username.strip()
            st.write(f"[DEBUG] Session state set. Triggering rerun.")
            st.rerun()
        else:
            st.error("Invalid credentials or missing username.")
            st.write(f"[DEBUG] Login failed. Credentials or username invalid.")

def header_bar():
    # MongoDB diagnostic block
    if st.session_state.get('role'):
        try:
            db = get_db()
            if db is not None:
                st.success("MongoDB connection test: Success!")
            else:
                st.error("MongoDB not connected.")
        except Exception as e:
            st.error(f"MongoDB connection test failed: {e}")
    left, mid, right = st.columns([0.25,0.5,0.25])
    with left:
        st.caption("Conversational Banking – Pre‑POC Discovery (v4)")
        st.write(f"**User:** {st.session_state.get('username','')} ({st.session_state.get('role','')})")
    with mid:
        # Connection status
        openai_status = "❌ OpenAI Not Connected"
        mongo_status = "❌ MongoDB Not Connected"
        try:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if OpenAI and api_key:
                client = OpenAI(api_key=api_key)
                models = client.models.list()
                if hasattr(models, "data") and len(models.data) > 0:
                    openai_status = "✅ OpenAI Connected"
        except Exception:
            pass
        try:
            db = get_db()
            if db is not None:
                mongo_status = "✅ MongoDB Connected"
        except Exception:
            pass
        st.write(openai_status)
        st.write(mongo_status)
    with right:
        st.write("")
        if st.button("Logout", help="Click to end your session."):
            for k in ["role","username","current_doc_id","fixed_answers","open_blocks"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

def is_yes_no_only(options: List[str]) -> bool:
    if not options:
        return False
    norm = [o.strip().lower() for o in options]
    return set(norm) == {"yes","no"} or set(norm) == {"no","yes"}

def help_bubble(text: str, key: str):
    # If no tooltip or too short, use OpenAI to generate a helpful tip
    if not text or len(text.strip()) < 10:
        # Try to generate a tip using OpenAI if available
        q = key.replace('fx_', '')
        question = f"Explain in simple, friendly language how someone should answer this banking survey question: '{q}'. Give practical tips and examples so anyone can understand what to write."
        tip = None
        try:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if OpenAI and api_key:
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}],
                    max_tokens=80,
                    temperature=0.3
                )
                tip = resp.choices[0].message.content.strip()
        except Exception:
            tip = None
        text = tip or "Provide clear, specific, and relevant details to help us understand your answer."
    # Prefer popover if available (Streamlit >= 1.32), else expander
    if hasattr(st, "popover"):
        with st.popover("ⓘ Details", use_container_width=False):
            st.write(text)
    else:
        with st.expander("ⓘ Details"):
            st.write(text)

def render_question(q: Dict[str, Any], key_prefix=""):
    qid = q["id"]
    label = f"{qid} — {q['text']}"
    help_txt = q.get("tooltip","")
    qtype = q.get("type","text")
    key = f"{key_prefix}{qid}"

    # Layout: label + bubble
    lbl_col, bubble_col = st.columns([0.92, 0.08])
    with lbl_col:
        st.markdown(f"**{label}**")
    with bubble_col:
        help_bubble(help_txt, key)

    # Auto multi-select for non-yes/no lists when type is 'select'
    if qtype == "select":
        opts = q.get("options", [])
        auto_multi = True if q.get("auto_multi", True) else False
        if auto_multi and opts and not is_yes_no_only(opts):
            qtype = "multiselect"

    # Render input
    if qtype == "text":
        ans = st.text_area("", key=key, height=80, placeholder="Type your answer here...")
        return ans

    elif qtype == "select":
        opts = q.get("options", [])
        ans = st.selectbox("", options=opts, key=key, index=0 if opts else None, placeholder="Select one...")
        # Others immediate free-text
        if isinstance(ans, str) and ans and ans.strip().lower() == "others":
            other = st.text_input("Please specify 'Others':", key=f"{key}_other")
            if other:
                ans = f"Others: {other}"
        return ans

    elif qtype == "multiselect":
        opts = q.get("options", [])
        selection = st.multiselect("", options=opts, key=key, default=[])
        # Others immediate free-text
        if any(isinstance(x, str) and x.strip().lower()=="others" for x in selection):
            other = st.text_input("Please specify 'Others':", key=f"{key}_other")
            if other:
                # replace 'Others' token with 'Others: ...'
                selection = [f"Others: {other}" if (isinstance(x,str) and x.strip().lower()=="others") else x for x in selection]
        return selection

    elif qtype == "likert":
        lk = q.get("likert", {"min":1,"max":5,"labels":["1","2","3","4","5"]})
        val = st.slider("", min_value=int(lk.get("min",1)), max_value=int(lk.get("max",5)), value=int(lk.get("min",1)), step=1, key=key)
        labels = lk.get("labels", [])
        if labels and len(labels) >= (lk.get("max",5)-lk.get("min",1)+1):
            st.caption(" | ".join([f"{i}:{t}" for i,t in enumerate(labels, start=lk.get('min',1))]))
        return val

    else:
        ans = st.text_area("", key=key, height=80, placeholder="Type your answer here...")
        return ans

def validate_required(questions: List[Dict[str,Any]], answers: Dict[str,Any]) -> List[str]:
    missing = []
    for q in questions:
        if not q.get("required", False):
            continue
        v = answers.get(q["id"])
        qtype = q.get("type","text")
        # if 'select' auto-morphed to 'multiselect', answers may be list
        if isinstance(v, list):
            if not v:
                missing.append(q["id"])
        else:
            if qtype in ["text"] and (not v or not str(v).strip()):
                missing.append(q["id"])
            elif qtype in ["select"] and (v is None or v == ""):
                missing.append(q["id"])
            elif qtype in ["likert"] and (v is None):
                missing.append(q["id"])
    return missing

def page_survey(cfg, role):
    header_bar()
    st.subheader("Step 1 — Fixed Questions")
    qs = get_questions(cfg)

    # Step 1 form captures but DOES NOT save to DB
    with st.form("fixed_form"):
        answers = {}
        for q in qs:
            ans = render_question(q, key_prefix="fx_")
            answers[q["id"]] = ans
        next_step = st.form_submit_button("Continue to Open‑Ended")
    if next_step:
        miss = validate_required(qs, answers)
        if miss:
            st.error("Please answer all required questions: " + ", ".join(miss))
        else:
            st.session_state["fixed_answers"] = [
                {"id": q["id"], "question": q["text"], "pillar": q["pillar"], "category": q["category"], "type": q["type"], "answer": answers.get(q["id"])}
                for q in qs
            ]
            st.success("Fixed questions captured. Now answer the Open‑Ended section below.")

    st.subheader("Step 2 — Open‑Ended + Dynamic Follow‑ups")
    num_open = int(cfg["APP"]["num_open_ended"])
    k_follow = int(cfg["APP"]["num_followups_per_open"])
    # Robustly parse open_ended_prompts from config
    open_prompts = []
    if "QUESTIONS" in cfg and "open_ended_prompts" in cfg["QUESTIONS"]:
        val = cfg["QUESTIONS"]["open_ended_prompts"]
        # Preprocess: convert single quotes to double quotes for JSON compatibility
        if val and "'" in val and '"' not in val:
            val = val.replace("'", '"')
        try:
            open_prompts = json.loads(val)
            if not isinstance(open_prompts, list):
                open_prompts = [str(open_prompts)]
        except Exception:
            open_prompts = [f"Describe area {i+1}..." for i in range(num_open)]
    else:
        open_prompts = [f"Describe area {i+1}..." for i in range(num_open)]

    open_blocks = []
    for i in range(num_open):
        with st.expander(f"Open‑ended {i+1}", expanded=(i==0)):
            prompt = open_prompts[i] if i < len(open_prompts) else f"Describe area {i+1}..."
            st.info(prompt, icon="💬")
            ans = st.text_area("Your answer", key=f"open_ans_{i}")
            gen = st.button(f"Generate {k_follow} follow‑ups", key=f"gen_{i}")
            if gen:
                sys_p = cfg["DYNAMIC_FOLLOWUPS"]["followup_system_prompt"]
                usr_t = cfg["DYNAMIC_FOLLOWUPS"]["followup_user_template"]
                model = cfg["OPENAI"]["model"]
                temp = float(cfg["OPENAI"]["temperature"]); max_t = int(cfg["OPENAI"]["max_tokens"])
                st.session_state[f"followups_{i}"] = openai_followups(k_follow, sys_p, usr_t, ans, model, temp, max_t)
            qs_f = st.session_state.get(f"followups_{i}", [])
            fu_answers = []
            if qs_f:
                st.write("Answer the follow‑ups:")
                for j, qf in enumerate(qs_f, 1):
                    a = st.text_input(f"F{j}: {qf}", key=f"fu_{i}_{j}")
                    fu_answers.append({"q": qf, "a": a})
            open_blocks.append({"prompt": prompt, "answer": ans, "followups": fu_answers})

    st.subheader("Step 3 — Submit & Analyze")
    # Final single submit after open-ended are answered
    if st.button("Submit Survey (Save to MongoDB)"):
        # Validate we captured Step 1 and open-ended
        fixed = st.session_state.get("fixed_answers", [])
        if not fixed:
            st.error("Please complete Step 1 (Fixed Questions) and click 'Continue to Open‑Ended' first.")
            return
        # Require open-ended answers
        missing_open = [i+1 for i, ob in enumerate(open_blocks) if not ob.get("answer","").strip()]
        if missing_open:
            st.error("Please answer all open‑ended questions: " + ", ".join([f"{i}" for i in missing_open]))
            return

        # Save to Mongo
        db = get_db()
        col = None
        if db is not None:
            col = db[cfg["MONGO"]["collection_name"]]

        org_col1, org_col2 = st.columns(2)
        with org_col1:
            org_name = st.text_input("Organization Name (for record)", key="org_name_submit", placeholder="Optional")
        with org_col2:
            contact = st.text_input("Contact (name/email)", key="org_contact_submit", placeholder="Optional")

        doc = {
            "org": {"name": org_name, "contact": contact},
            "answers": {"fixed": fixed, "open": open_blocks},
            "status": "submitted",
            "submitted_by": st.session_state.get("username",""),
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if col is not None:
            try:
                res = col.insert_one(doc)
                st.session_state["current_doc_id"] = str(res.inserted_id)
                st.success("Survey saved to MongoDB.")
            except Exception as e:
                import pymongo
                if isinstance(e, pymongo.errors.ServerSelectionTimeoutError):
                    st.error("MongoDB server is unreachable during insert. Please check your network, URI, and Atlas cluster status.")
                else:
                    st.error(f"MongoDB insert error: {e}")
        else:
            st.info("MONGO_URI not set or pymongo missing — skipped DB save.")

        # PDF report generation and download link for user
        import io
        from fpdf import FPDF
        import base64
        # Simple scorer with broadened pillars (same as admin)
        seeds = {
            "Business & Strategic Alignment": ["kpi","csat","nps","journey","omni","target","conversion"],
            "Scope & Use Cases": ["intent","journey","transfer","transaction","multilingual","language"],
            "Technology & Integration": ["api","middleware","sso","otp","biometric","whatsapp","ivr"],
            "Risk, Governance & Operations": ["handoff","sla","monitor","feedback","bias","fairness","ethics","content"],
            "Infrastructure, AI Readiness & Security": ["gpu","h100","a100","mlops","databricks","sagemaker","vertex","gateway","apigee","kong","mulesoft","prometheus","grafana","elastic","dr","ha","sandbox"],
            "Model & Platform": ["openai","azure","anthropic","cohere","dbrx","llama","embedding","fine-tune"],
            "Validation & Testing": ["eval","dataset","metrics","red-team","test","qa","sign-off","sandbox"]
        }
        text_blob = []
        for f in fixed:
            text_blob.append(str(f.get("answer","")))
        for op in open_blocks:
            text_blob.append(op.get("answer",""))
            for fu in op.get("followups", []):
                text_blob.append(fu.get("a",""))
        full = " ".join(text_blob).lower()

        pillars = []
        total = 0
        for name, kws in seeds.items():
            hits = sum(1 for kw in kws if kw in full)
            score = min(1 + hits*2, 20)
            total += score
            stage = "Nascent"
            for thr, lab in [(1,"Nascent"),(5,"Emerging"),(10,"Developing"),(15,"Advanced"),(20,"Leading")]:
                if score >= thr: stage = lab
            pillars.append({"name": name, "score": score, "stage": stage})
        sc = {"pillars": pillars, "overall": total}

        import unicodedata
        def to_ascii(text):
            return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, to_ascii("Conversational Banking Discovery Insights Report"), ln=True, align="C")
        pdf.ln(5)
        pdf.cell(0, 10, to_ascii(f"Overall Score: {sc['overall']}"), ln=True)
        pdf.cell(0, 10, to_ascii("Pillar Insights:"), ln=True)
        for p in sc["pillars"]:
            pdf.cell(0, 10, to_ascii(f"- {p['name']}: {p['score']} ({p['stage']})"), ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, to_ascii("Recommended Next Steps:"), ln=True)
        # Optionally load next steps from scoring_rules.json if available
        import os, json
        scoring_path = os.path.join(os.path.dirname(__file__), "scoring_rules.json")
        next_steps = {}
        try:
            with open(scoring_path, "r", encoding="utf-8") as f:
                scoring = json.load(f)
                next_steps = scoring.get("next_steps", {})
        except Exception:
            next_steps = {}
        for p in sc["pillars"]:
            steps = next_steps.get(p["name"], [])
            if steps:
                pdf.cell(0, 10, to_ascii(f"{p['name']}:"), ln=True)
                for s in steps:
                    pdf.cell(0, 10, to_ascii(f"- {s}"), ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, to_ascii("Discrepancies Found:"), ln=True)
        # Simple discrepancy check (missing/short answers)
        discrepancy_summary = []
        for q in fixed:
            ans = q.get("answer","")
            if not ans or len(str(ans).strip()) < 3:
                discrepancy_summary.append(f"Missing or too short answer for question: {q.get('question','')}")
        for op in open_blocks:
            if not op.get("answer","") or len(str(op.get("answer"," ")).strip()) < 3:
                discrepancy_summary.append(f"Missing or too short open-ended answer: {op.get('prompt','')}")
        if discrepancy_summary:
            for issue in discrepancy_summary:
                pdf.cell(0, 10, to_ascii(f"- {issue}"), ln=True)
        else:
            pdf.cell(0, 10, to_ascii("No discrepancies found in analyzed record."), ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, to_ascii("How Scores Are Calculated:"), ln=True)
        pdf.multi_cell(0, 10, to_ascii("Scores are calculated by scanning all answers for pillar-specific keywords. Each keyword hit adds points to the relevant pillar, with a base score of 1 per pillar. The total score per pillar is capped at 20. Pillar stages are assigned based on thresholds: Nascent (1+), Emerging (5+), Developing (10+), Advanced (15+), Leading (20). The overall score is the sum of all pillar scores."))
        import unicodedata
        def to_ascii(text):
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        # Use to_ascii for all text added to PDF
        # Example: pdf.cell(0, 10, to_ascii("text"), ln=True)
        # If you want to clean all text, wrap all pdf.cell/multi_cell calls with to_ascii()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="CB_Discovery_Insights_Report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Optional — Compute Maturity & Save Report")
    if st.button("Compute & Save Report"):
        # Build from current session state
        fixed_list = st.session_state.get("fixed_answers", [])
        if not fixed_list:
            st.error("Please complete Step 1 and Step 3 submission first.")
            return
        op_blocks = []
        for i in range(int(cfg["APP"]["num_open_ended"])):
            ops = st.session_state.get(f"followups_{i}", [])
            fu = []
            for j, _ in enumerate(ops, 1):
                fu.append({"q": st.session_state.get(f"fu_{i}_{j-0}", ""), "a": st.session_state.get(f"fu_{i}_{j}", "")})
            op_blocks.append({"prompt": "", "answer": st.session_state.get(f"open_ans_{i}", ""), "followups": fu})

        # Simple scorer with broadened pillars
        seeds = {
            "Business & Strategic Alignment": ["kpi","csat","nps","journey","omni","target","conversion"],
            "Scope & Use Cases": ["intent","journey","transfer","transaction","multilingual","language"],
            "Technology & Integration": ["api","middleware","sso","otp","biometric","whatsapp","ivr"],
            "Risk, Governance & Operations": ["handoff","sla","monitor","feedback","bias","fairness","ethics","content"],
            "Infrastructure, AI Readiness & Security": ["gpu","h100","a100","mlops","databricks","sagemaker","vertex","gateway","apigee","kong","mulesoft","prometheus","grafana","elastic","dr","ha","sandbox"],
            "Model & Platform": ["openai","azure","anthropic","cohere","dbrx","llama","embedding","fine-tune"],
            "Validation & Testing": ["eval","dataset","metrics","red-team","test","qa","sign-off","sandbox"]
        }
        text_blob = []
        for f in fixed_list:
            text_blob.append(str(f.get("answer","")))
        for op in op_blocks:
            text_blob.append(op.get("answer",""))
            for fu in op.get("followups", []):
                text_blob.append(fu.get("a",""))
        full = " ".join(text_blob).lower()

        pillars = []
        total = 0
        for name, kws in seeds.items():
            hits = sum(1 for kw in kws if kw in full)
            score = min(1 + hits*2, 20)
            total += score
            stage = "Nascent"
            for thr, lab in [(1,"Nascent"),(5,"Emerging"),(10,"Developing"),(15,"Advanced"),(20,"Leading")]:
                if score >= thr: stage = lab
            pillars.append({"name": name, "score": score, "stage": stage})
        sc = {"pillars": pillars, "overall": total}

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"CB_Discovery_Report_{ts}.md"
    if 'sc' in locals():
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Conversational Banking Pre‑POC Discovery — Results\n\n")
            f.write("## Summary Scores\n")
            for p in sc["pillars"]:
                f.write(f"- **{p['name']}** — {p['score']} ({p['stage']})\n")
            f.write(f"\n**Overall:** {sc['overall']}\n")
        st.success(f"Report saved: {path}")
        st.markdown(f"[Download the report]({path})")
    else:
        st.warning("No scores available. Report not generated.")

    # Save to Mongo if we have a current_doc_id and scores
    if "current_doc_id" in st.session_state and get_db() is not None and 'sc' in locals():
        from bson import ObjectId
        db = get_db()
        col = db[cfg["MONGO"]["collection_name"]]
        col.update_one({"_id": ObjectId(st.session_state["current_doc_id"])}, {"$set": {"scores": sc, "status":"analyzed"}})
        st.toast("Scores saved to MongoDB.")

def page_admin(cfg):
    # --- Admin page header with Logout button ---
    header_col1, header_col2 = st.columns([0.8, 0.2])
    with header_col1:
        st.caption("Conversational Banking – Admin Console (v4)")
    with header_col2:
        if st.button("Logout", help="Click to end your session.", key="admin_logout_btn"):
            for k in ["role","username","current_doc_id","fixed_answers","open_blocks"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
    db = get_db()
    col = None
    collection_name = cfg["MONGO"].get("collection_name", "")
    if db is not None and collection_name:
        try:
            col = db[collection_name]
        except Exception as e:
            st.error(f"Could not access collection '{collection_name}': {e}")
    elif not collection_name:
        st.error("MongoDB collection name is missing in config.")
    # Show actual collection names in the connected database for debugging
    sel = ""
    def to_ascii(text):
        return str(text).encode('ascii', 'ignore').decode()

    # --- Restore full admin features ---
    def _header():
        left, mid, right = st.columns([0.25,0.5,0.25])
        with left:
            st.caption("Conversational Banking – Admin Console (v4)")
        with right:
            if st.button("Logout", help="Click to end your session.", key="admin_logout_btn"):
                for k in ["role","username","current_doc_id","fixed_answers","open_blocks"]:
                    if db is None:
                        st.info("MONGO_URI not set or pymongo missing — admin features disabled.")
                    else:
                        if cfg and "MONGO" in cfg and "collection_name" in cfg["MONGO"]:
                            col = db[cfg["MONGO"]["collection_name"]]
                        else:
                            col = None
                        # Use same connection test as user role
                        if st.button("Test Mongo Connectivity", key="admin_test_mongo_connectivity"):
                            if db is not None:
                                st.success("MongoDB connection test: Success!")
                            else:
                                st.error("MongoDB not connected.")
                        with st.expander("Filters"):
                            org = st.text_input("Organization contains")
                            submitter = st.text_input("Submitted by contains")
                            status = st.multiselect("Status", ["submitted","analyzed"], default=[])
                            limit = st.number_input("Max records", 1, 1000, 100)

                        query = {}
                        if org: query["org.name"] = {"$regex": org, "$options":"i"}
                        if submitter: query["submitted_by"] = {"$regex": submitter, "$options":"i"}
                        if status: query["status"] = {"$in": status}

                    # Only define and use query/limit once for filtering
                    query = {}
                    org = st.text_input("Organization contains", key="admin_org_filter")
                    submitter = st.text_input("Submitted by contains", key="admin_submitter_filter")
                    status = st.multiselect("Status", ["submitted","analyzed"], default=[], key="admin_status_filter")
                    limit = st.number_input("Max records", 1, 1000, 100, key="admin_limit_filter")
                    if org: query["org.name"] = {"$regex": org, "$options":"i"}
                    if submitter: query["submitted_by"] = {"$regex": submitter, "$options":"i"}
                    if status: query["status"] = {"$in": status}
                    try:
                        rows = list(col.find(query).sort("created_at",-1).limit(int(limit)))
                    except Exception as e:
                        import pymongo
                        if isinstance(e, pymongo.errors.ServerSelectionTimeoutError):
                            st.error("MongoDB server is unreachable. Please check your network, URI, and Atlas cluster status.")
                            rows = []
                        else:
                            st.error(f"MongoDB error: {e}")
                            rows = []

    # --- Restore tabs: Records & Insights and Admin Settings ---
    tabs = ["Records & Insights"]
    if st.session_state.get("role") == "Admin":
        tabs.append("Admin Settings")
    tab_objs = st.tabs(tabs)

    # --- Filters and query definition ---
    org = st.text_input("Organization contains", key="admin_org_filter")
    submitter = st.text_input("Submitted by contains", key="admin_submitter_filter")
    status = st.multiselect("Status", ["submitted","analyzed"], default=[], key="admin_status_filter")
    limit = st.number_input("Max records", 1, 1000, 100, key="admin_limit_filter")
    query = {}
    if org: query["org.name"] = {"$regex": org, "$options":"i"}
    if submitter: query["submitted_by"] = {"$regex": submitter, "$options":"i"}
    if status: query["status"] = {"$in": status}
    rows = []
    if col is None:
        st.error("MongoDB collection is not available. Please check your configuration and connection.")
    else:
        try:
            rows = list(col.find(query).sort("created_at",-1).limit(int(limit)))
        except Exception as e:
            import pymongo
            if isinstance(e, pymongo.errors.ServerSelectionTimeoutError):
                st.error("MongoDB server is unreachable. Please check your network, URI, and Atlas cluster status.")
                rows = []
            else:
                st.error(f"MongoDB error: {e}")
                rows = []

    # Records & Insights tab
    with tab_objs[tabs.index("Records & Insights")]:
        sel = ""
        if rows and isinstance(rows, list) and len(rows) > 0:
            df = pd.DataFrame([
                {
                    "id": str(r.get("_id")),
                    "org": (r.get("org") or {}).get("name", ""),
                    "submitted_by": r.get("submitted_by", ""),
                    "status": r.get("status", ""),
                    "created_at": r.get("created_at", "")
                }
                for r in rows
            ])
            # st.dataframe(df, use_container_width=True)
            # sel = st.selectbox("Open record", options=[""] + df["id"].tolist(), key="admin_open_record_selectbox")
        if sel:
            doc = col.find_one({"_id": ObjectId(sel)})
            st.json(doc)

            if st.button("Compute Scores (if missing)", key=f"compute_scores_{sel}"):
                answers = doc.get("answers", {})
                text_blob = []
                for f in answers.get("fixed", []): text_blob.append(str(f.get("answer","")))
                for op in answers.get("open", []):
                    text_blob.append(op.get("answer",""))
                    for fu in op.get("followups", []):
                        text_blob.append(fu.get("a",""))
                full = " ".join(text_blob).lower()
                seeds = {
                    "Business & Strategic Alignment": ["kpi","csat","journey","omni","nps","target","conversion"],
                    "Scope & Use Cases": ["intent","journey","transfer","transaction","multilingual","language"],
                    "Technology & Integration": ["api","middleware","sso","otp","biometric","whatsapp","ivr"],
                    "Risk, Governance & Operations": ["handoff","sla","monitor","feedback","bias","fairness","ethics","content"],
                    "Infrastructure, AI Readiness & Security": ["gpu","h100","a100","mlops","databricks","sagemaker","vertex","gateway","apigee","kong","mulesoft","prometheus","grafana","elastic","dr","ha","sandbox"],
                    "Model & Platform": ["openai","azure","anthropic","cohere","dbrx","llama","embedding","fine-tune"],
                    "Validation & Testing": ["eval","dataset","metrics","red-team","test","qa","sign-off","sandbox"]
                }
                pillars = []
                total = 0
                for name, kws in seeds.items():
                    hits = sum(1 for kw in kws if kw in full)
                    score = min(1 + hits*2, 20)
                    total += score
                    stage = "Nascent"
                    for thr, lab in [(1,"Nascent"),(5,"Emerging"),(10,"Developing"),(15,"Advanced"),(20,"Leading")]:
                        if score >= thr: stage = lab
                    pillars.append({"name": name, "score": score, "stage": stage})
                sc = {"pillars": pillars, "overall": total}
                col.update_one({"_id": ObjectId(sel)}, {"$set":{"scores": sc, "status":"analyzed"}})
                st.success("Scores computed and saved.")
                st.markdown("---")
                doc = col.find_one({"_id": ObjectId(sel)})
                st.json(doc)

            # Always show Discrepancy Check after record selection and score computation
            st.subheader("Discrepancy Check")
            fixed_answers = doc.get("answers",{}).get("fixed", [])
            open_answers = doc.get("answers",{}).get("open", [])
            all_answers = [(q.get("question",""), str(q.get("answer",""))) for q in fixed_answers] + [(op.get("prompt",""), str(op.get("answer",""))) for op in open_answers]

            # Prepare prompt for OpenAI
            prompt = """
            You are an expert survey analyst. Given the following questions and answers from a banking discovery survey, analyze for discrepancies, contradictions, or incomplete responses. For each issue, list:
            1. The question(s)
            2. The answer(s)
            3. The discrepancy or issue found
            4. Suggestions to resolve or clarify

            Only report issues that are clear, significant, or could impact survey validity. Be concise and specific.
            """

    # Admin Settings tab
    if "Admin Settings" in tabs:
        with tab_objs[tabs.index("Admin Settings")]:
            st.subheader("Admin Settings")
            st.markdown("### Edit Fixed Questions")
            import json
            questions = json.loads(cfg["QUESTIONS"]["questions_json"])
            edited_questions = []
            with st.form("admin_edit_questions_form"):
                for q in questions:
                    st.markdown(f"**{q['id']}** — {q['text']}")
                    new_text = st.text_area("Question Text", value=q['text'], key=f"admin_qtext_{q['id']}")
                    new_type = st.selectbox("Type", options=["text","select","multiselect","likert"], index=["text","select","multiselect","likert"].index(q.get("type","text")), key=f"admin_qtype_{q['id']}")
                    new_category = st.text_input("Category", value=q.get("category",""), key=f"admin_qcat_{q['id']}")
                    new_required = st.checkbox("Required", value=q.get("required",False), key=f"admin_qreq_{q['id']}")
                    edited_questions.append({**q, "text": new_text, "type": new_type, "category": new_category, "required": new_required})
                save_btn = st.form_submit_button("Save Changes")
            if save_btn:
                cfg["QUESTIONS"]["questions_json"] = json.dumps(edited_questions, ensure_ascii=False)
                st.success("Questions updated in config (in-memory only; restart to persist).")

            st.markdown("---")
            st.markdown("### Prompts for Open-Ended Questions")
            open_prompts = json.loads(cfg["QUESTIONS"].get("open_ended_prompts","[]"))
            for i, p in enumerate(open_prompts, 1):
                st.text_area(f"Prompt {i}", value=p, key=f"admin_open_prompt_{i}")

            st.markdown("---")
            st.markdown("### Dynamic Followups Prompts")
            st.text_area("Followup System Prompt", value=cfg["DYNAMIC_FOLLOWUPS"].get("followup_system_prompt",""), key="admin_followup_system_prompt")
            st.text_area("Followup User Template", value=cfg["DYNAMIC_FOLLOWUPS"].get("followup_user_template",""), key="admin_followup_user_template")

        # Records & Insights tab
        with tab_objs[tabs.index("Records & Insights")]:
            try:
                rows = list(col.find(query).sort("created_at",-1).limit(int(limit)))
            except Exception as e:
                import pymongo
                if isinstance(e, pymongo.errors.ServerSelectionTimeoutError):
                    st.error("MongoDB server is unreachable. Please check your network, URI, and Atlas cluster status.")
                    rows = []
                else:
                    st.error(f"MongoDB error: {e}")
                    rows = []
            sel = ""
            if rows:
                df = pd.DataFrame([
                    {
                        "id": str(r.get("_id")),
                        "org": (r.get("org") or {}).get("name", ""),
                        "submitted_by": r.get("submitted_by", ""),
                        "status": r.get("status", ""),
                        "created_at": r.get("created_at", "")
                    }
                    for r in rows
                ])
                st.dataframe(df, use_container_width=True)
                sel = st.selectbox("Open record", options=[""] + df["id"].tolist())
            if sel:
                doc = col.find_one({"_id": ObjectId(sel)})
                st.json(doc)
                # ...existing code for record details, scores, discrepancy check, etc...

        # Admin Settings tab
        if "Admin Settings" in tabs:
            with tab_objs[tabs.index("Admin Settings")]:
                st.subheader("Admin Settings")
                st.info("Admin settings features placeholder. Add your settings management here.")

        # Insights & Next Steps Section
        st.subheader("Insights & Next Steps")
        if rows:
            # Find the most recent analyzed record with scores
            analyzed = [r for r in rows if r.get("scores")]
            if analyzed:
                latest = analyzed[0]
                scores = latest["scores"]
                pillars = scores.get("pillars", [])
                overall = scores.get("overall", 0)
                st.markdown(f"**Overall Score:** {overall}")
                st.markdown("### Pillar Insights:")
                for p in pillars:
                    st.markdown(f"- **{p['name']}**: {p['score']} ({p['stage']})")
                st.markdown("---")
                st.markdown("#### How Overall Score is Calculated")
                st.info("""
**Score Calculation Logic:**
- For each pillar, answers are scanned for pillar-specific keywords.
- Each keyword hit adds 2 points to the pillar, with a base score of 1 per pillar.
- The total score per pillar is capped at 20.
- Pillar stages are assigned based on thresholds: Nascent (1+), Emerging (5+), Developing (10+), Advanced (15+), Leading (20).
- The **Overall Score** is the sum of all pillar scores.
                """)
                # Next Steps from scoring_rules.json if available
                scoring_path = os.path.join(os.path.dirname(__file__), "scoring_rules.json")
                next_steps = {}
                try:
                    with open(scoring_path, "r", encoding="utf-8") as f:
                        scoring = json.load(f)
                        next_steps = scoring.get("next_steps", {})
                except Exception:
                    next_steps = {}
                st.markdown("### Recommended Next Steps:")
                for p in pillars:
                    steps = next_steps.get(p["name"], [])
                    if steps:
                        st.markdown(f"**{p['name']}**:")
                        for s in steps:
                            st.markdown(f"- {s}")
            else:
                st.info("No analyzed records with scores found for insights.")

# --- MAIN PAGE ROUTING ---
if __name__ == "__main__":
    cfg = load_cfg()
    # Show MongoDB and OpenAI connection status at the top
    db = get_db()
    mongo_status = "✅ MongoDB Connected" if db is not None else "❌ MongoDB Not Connected"
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        openai_status = "❌ OpenAI Not Connected"
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            if hasattr(models, "data") and len(models.data) > 0:
                openai_status = "✅ OpenAI Connected"
    except Exception:
        openai_status = "❌ OpenAI Not Connected"
    st.markdown(f"**MongoDB Status:** {mongo_status}")
    st.markdown(f"**OpenAI Status:** {openai_status}")

    if not st.session_state.get("role"):
        login_screen(cfg)
    else:
        role = st.session_state.get("role")
        if role == "Admin":
            page_admin(cfg)
        elif role in ["User", "Head", "Data Infrastructure"]:
            page_survey(cfg, role)
        else:
            st.error("Unknown role. Please login again.")
