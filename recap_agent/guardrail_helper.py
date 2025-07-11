from guardrails import Guard
from guardrails.hub import ToxicLanguage
from typing import Any
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

# Initialize toxicity guard (install torch first: pip install torch)
toxicity_guard = Guard().use_many(
    ToxicLanguage(
        threshold=0.3,  # 0-1 (higher = more sensitive)
        validation_method="sentence",
        on_fail="exception"
    )
)

def guard_input(llm_request, callback_context) -> Any:
    """Check user input for toxicity"""
    if not llm_request.contents:
        return None
        
    last_content = llm_request.contents[-1]
    if last_content.role != "user" or not last_content.parts:
        return None
        
    user_text = last_content.parts[0].text
    if not user_text:
        return None
    
    try:
        toxicity_guard.validate(user_text)
        print("user_text --------> ", user_text)
    except Exception:
        return LlmResponse(
            content=Content(
                role='model',
                parts=[Part(text="Let's keep our conversation respectful. Please rephrase your message.")]
            )
        )
    return None

def guard_output(llm_response, callback_context) -> Any:
    """Check AI responses for toxicity before sending"""
    if not llm_response.content:
        return llm_response
        
    model_content = llm_response.content
    if model_content.role != "model" or not model_content.parts:
        return llm_response
        
    model_text = model_content.parts[0].text
    if not model_text:
        return llm_response
    
    try:
        toxicity_guard.validate(model_text)
    except Exception:
        return LlmResponse(
            content=Content(
                role='model',
                parts=[Part(text="I can't provide that response. Let's focus on positive learning!")]
            )
        )
    return llm_response