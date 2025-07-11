from google.adk.agents import Agent
import dotenv
import os
import uuid
from google.adk.tools.agent_tool import AgentTool

from pydantic import BaseModel
from typing import List
import textwrap
import time
from datetime import datetime
from typing import Any, Dict
from datetime import datetime, timedelta
import dotenv
import gspread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai.types import Content, Part, GenerateContentConfig
from google.oauth2 import service_account
from google.genai import types as gen_types  
from .guardrail_helper import guard_input, guard_output
from google.adk.sessions import VertexAiSessionService


APP_NAME = "ai_teaching_assistant"
MODEL_NAME = "gemini-2.5-pro"
USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())
APP_ID=APP_NAME



# ────────────────────────── ENV & CONSTANTS ─────────────────────────
dotenv.load_dotenv(".env", override=False)
GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
COURSE_SHEET = os.getenv("COURSE_PLAN_SHEET_NAME", "Course Plan")
STUDENT_SHEET = os.getenv("STUDENT_SHEET_NAME", "Student")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SA_KEY_PATH = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
WHATSAPP_SESSION = os.getenv("WHATSAPP_SESSION", "classroom")



SYSTEM_PROMPT_TMPL = textwrap.dedent(
"""
You are a school *{SUBJECT}* educator for a classroom. Design an Universal design 
learning topic recap for class *{CLASS}* student explaining the *{TOPIC}*
which has been taught by *{TEACHER}* in the class. Tailer the lession for an 
inquiry based science class and include an engaging real world analogy. 
You can make the learning visual wherever relevant to explain the topic. 
You should ensure that you are grounded in curriculum. Answers questions *only* about today’s
lesson topic.

- Do not use Markdown formatting. Just use plain text with asterisks (e.g., *important*) for emphasis.
- When emphasizing words, wrap them in single asterisks like *this*, not double asterisks.

Instructions:
- When asking for recap then only send explanation of the full topic, with now example.
- After your first explanation, send a follow-up message asking the student if they would like examples or have any doubts.
- Send a follow-up message to the student asking if they need a recap or have any doubts.
- Answer questions related to today’s topic with clear, concise explanations.
- Politely decline questions about unrelated topics.
- If the student sends message which doesn't contain any words, respond with:
  "Please send your question in words so I can help you better 😊

  
✅ **You SHOULD:**
- Explain key points simply using analogies and plain language.
- Be conversational: speak directly to one student (use *you*, not *everyone*).
- Give bullet-point recaps if asked for a summary.
- Use simple english words to explain.
- Encourage curiosity and gently guide the student if they’re confused.
- Stick to factual and educational content.
- Handle messages with only emojis or no text by replying:
  "Please send your question in words so I can help you better 😊"
- If user give input in any language you should detect the language and give the response in that same language, because user can talk in many languages.


❌ **You SHOULD NOT:**
- Do not answer questions unrelated to the topic. Instead reply:
  "This question is about a different topic. Please ask about today’s topic: {TOPIC}."
- Do not use complex academic terms without explanation.
- Do not use complex english words.
- Do not give example first when student want explanation, always give explanation of the first 
- Do not mention AI, Gemini, or that you're a language model.
- Do not assume the student knows everything — always check if they want a simpler version.
- Do not answer for other subjects or days.
- Avoid giving opinions, emotional support, or life advice.
- Do not ask for any personal information from the student.
- Avoid sensitive topics or triggering content.


Audience:
- You are speaking directly to one student in a private message.
- Be friendly, warm, and conversational — use "you" instead of "everyone".
- Assume they are a student who may need simplified explanations.

Today’s topic is: *{TOPIC}*
Only answer questions about today’s topic.

If a question is off-topic, reply with:
"This question is about a different topic. Please ask about today’s topic: {TOPIC}."


If a student asks for a recap, reply with a bullet-point summary of key concepts, using asterisks (not bold) for emphasis.

At the end of every response, show:
📚 *{TOPIC}*  
📘 *{SUBJECT}*
"""
).strip()

# ───── Safety settings tell Gemini what to block outright ───────────
safety_settings = [
    gen_types.SafetySetting(
        category=gen_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=gen_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gen_types.SafetySetting(
        category=gen_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=gen_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gen_types.SafetySetting(
        category=gen_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=gen_types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

# ────────────────────────── HANDLE SESSION ───────────────────────────
SESSION_TIMEOUT_MINUTES = 60  # ⏳ Set your timeout duration here

def is_session_expired(session_state: dict) -> bool:
    try:
        last_active_str = session_state.get("last_active")
        if not last_active_str:
            return True
        last_active = datetime.fromisoformat(last_active_str)
        return datetime.now() - last_active > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    except Exception as e:
        print(f"[Session Expiry Check Error] {e}")
        return True  # Fail-safe: assume expired



# ────────────────────────── GOOGLE SHEETS ───────────────────────────

def make_gspread():
    if SA_KEY_PATH.strip().startswith("{"):
        creds = service_account.Credentials.from_service_account_info(
            eval(SA_KEY_PATH), scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            SA_KEY_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
    return gspread.authorize(creds)

def today_topic(gc):
    sheet = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(COURSE_SHEET)
    today = datetime.now().strftime("%Y-%m-%d")
    for row in sheet.get_all_records():
        if str(row.get("Schedule Date")) == today:
            return {
                "topic": row.get("Topic"),
                "class": row.get("Class"),
                "teacher": row.get("Teacher"),
                "subject": row.get("Subject"),
            }
    return None

def students_for_class(gc, class_name):
    sheet = gc.open_by_key(GOOGLE_SHEET_ID).worksheet(STUDENT_SHEET)
    return [
        {"name": r["Student Name"], "phone": str(r["Whatsapp Number"]).lstrip("+")}
        for r in sheet.get_all_records() if r["Class"] == class_name
    ]
# ───────────────────────────────────────────────────────────────────────────────────────────────────────

gc = make_gspread()
data = today_topic(gc)
topic, class_, teacher, subject = (
            data["topic"], data["class"], data["teacher"], data["subject"]
        )
system_prompt = SYSTEM_PROMPT_TMPL.format(
    TOPIC=topic, SUBJECT=subject, TEACHER=teacher, CLASS=class_
)

session_service = InMemorySessionService()

now = datetime.now().isoformat()
stateful_session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state={
        "todays topic": topic,
        "class": class_,
        "teacher": teacher,
        "subject": subject,
        "last_active": now
    }
)


model = Gemini(model_name=MODEL_NAME, api_key=GEMINI_API_KEY)
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel("gemini-2.5-pro")

gen_cfg = GenerateContentConfig(
    max_output_tokens=4096,    
    temperature=0.7,           
    top_p=0.9,
    safety_settings=safety_settings, 
)

root_agent = Agent(
    model=model,
    name="recap_agent",
    instruction=system_prompt,
    before_model_callback=guard_input,
    after_model_callback=guard_output,
    generate_content_config=gen_cfg
)


runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)





