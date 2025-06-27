import pexpect
import requests
import re
import os
import time
import subprocess
import logging
import shlex
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ConversationHandler,
    JobQueue
)
from dotenv import load_dotenv
import asyncio
import pytz

# Load environment variables from .env file
load_dotenv()

# === Global Configuration ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Configure LLM models for various tasks
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "moonshotai/kimi-dev-72b:free")
ERROR_FIX_MODEL = os.getenv("ERROR_FIX_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free")
CONVERSATION_MODEL = os.getenv("CONVERSATION_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
COMMAND_CONVERSION_MODEL = os.getenv("COMMAND_CONVERSION_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free")
FILENAME_GEN_MODEL = os.getenv("FILENAME_GEN_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
INTENT_DETECTION_MODEL = os.getenv("INTENT_DETECTION_MODEL", "mistralai/mistral-small-3.2-24b-instruct")

# ANSI colors for Termux console output (for internal logs only)
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_PURPLE = "\033[95m"
COLOR_RESET = "\033[0m"

# State for ConversationHandler (debugging)
DEBUGGING_STATE = 1

# --- Logging Configuration ---
# Modify logger format to be more concise and readable
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Variables for System Information ---
SYSTEM_INFO = {
    "os": "Unknown",
    "shell": "Unknown",
    "neofetch_output": "Not available"
}

# --- Global Functions for Context Storage ---
user_contexts: dict = {}
chat_histories: dict = {}

def get_user_context(chat_id: int) -> dict:
    """Retrieves user context. Initializes if not already present."""
    if chat_id not in user_contexts:
        user_contexts[chat_id] = {
            "last_error_log": None,
            "last_command_run": None,
            "last_generated_code": None,
            "awaiting_debug_response": False,
            "full_error_output": [],
            "last_user_message_intent": None,
            "last_ai_response_type": None,
            "last_generated_code_language": None
        }
    return user_contexts[chat_id]

def get_chat_history(chat_id: int) -> list:
    """Retrieves user chat history. Initializes if not already present."""
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]

def _escape_plaintext_markdown_v2(text: str) -> str:
    """
    Escapes MarkdownV2 special characters in plain text (non-code),
    but specifically DOES NOT escape '*' and '_' as they are assumed
    to be used for bold and italic formatting.
    This function is for text segments that are NOT code blocks.
    """
    # Characters that must be escaped when appearing literally in MarkdownV2.
    # This list DOES NOT include '*' and '_', as we want them
    # to be interpreted as formatting.
    chars_to_escape_regex = r'[\[\]()~`>#+\-=|{}.!]' 

    # Escape literal backslash first, so it doesn't interfere with subsequent escapes
    text = text.replace('\\', '\\\\')

    # Use re.sub to escape specific characters
    escaped_text = re.sub(chars_to_escape_regex, r'\\\g<0>', text)
    
    return escaped_text

# === General Function: Call LLM ===
def call_llm(messages: list, model: str, api_key: str, max_tokens: int = 512, temperature: float = 0.7, extra_headers: dict = None) -> tuple[bool, str]:
    """
    Generic function to send requests to an LLM model (OpenRouter).
    Returns a tuple: (True, result) on success, (False, error_message) on failure.
    """
    if not api_key or not LLM_BASE_URL:
        logger.error("[LLM ERROR] API Key or LLM Base URL not set.")
        return False, "API Key or LLM Base URL not set. Please check configuration."

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        res = requests.post(LLM_BASE_URL, json=payload, headers=headers, timeout=300)
        res.raise_for_status()
        data = res.json()
        if "choices" in data and data["choices"]:
            return True, data["choices"][0]["message"]["content"]
        else:
            logger.error(f"[LLM] LLM response does not contain 'choices'. Debug response: {data}")
            return False, f"LLM response not in expected format. Debug response: {data}"
    except requests.exceptions.Timeout:
        logger.error(f"[LLM] LLM API request timed out ({LLM_BASE_URL}).")
        return False, f"LLM API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"[LLM] Failed to connect to LLM API ({LLM_BASE_URL}): {e}")
        return False, f"Failed to connect to LLM API: {e}"
    except KeyError as e:
        logger.error(f"[LLM] LLM response not in expected format (no 'choices' or 'message'): {e}. Debug response: {data}")
        return False, f"LLM response not in expected format: {e}. Debug response: {data}"
    except Exception as e:
        logger.error(f"[LLM] An unexpected error occurred while calling LLM: {e}")
        return False, f"An unexpected error occurred while calling LLM: {e}"

# === Function: Extract Code from LLM Response ===
def ekstrak_kode_dari_llm(text_response: str, target_language: str = None) -> tuple[str, str]:
    """
    Extracts Markdown code blocks from LLM responses.
    Returns a tuple: (cleaned_code, detected_language)
    """
    code_block_pattern = r"```(?P<lang>\w+)?\n(?P<content>.*?)```"
    matches = re.findall(code_block_pattern, text_response, re.DOTALL)
    
    if matches:
        if target_language:
            for lang, content in matches:
                if lang and lang.lower() == target_language.lower():
                    logger.info(f"{COLOR_GREEN}[LLM] ✔ {target_language} code detected and extracted.{COLOR_RESET}")
                    return content.strip(), lang.lower()
        
        for lang, content in matches:
            if lang and lang.lower() in ["python", "bash", "javascript", "js", "sh", "py", "node"]:
                logger.info(f"{COLOR_GREEN}[LLM] ✔ {lang} code detected and extracted.{COLOR_RESET}")
                return content.strip(), lang.lower().replace("js", "javascript").replace("sh", "bash").replace("py", "python")
        
        if matches:
            logger.info(f"{COLOR_YELLOW}[LLM] ⚠ Markdown code block found without specific language indicator. Extracting and trying to guess language.{COLOR_RESET}")
            first_content = matches[0][1].strip()
            detected_lang = deteksi_bahasa_pemrograman_dari_konten(first_content)
            return first_content, detected_lang
    
    logger.warning(f"{COLOR_YELLOW}[LLM] ⚠ No Markdown code block detected. Performing aggressive text cleaning.{COLOR_RESET}")
    lines = text_response.strip().split('\n')
    cleaned_lines = []
    in_potential_code_block = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(('#', 'import ', 'from ', 'function ', 'def ', 'const ', 'let ', 'var ', 'echo ', '#!/')):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif re.match(r'^(def|class|if|for|while|try|with|function|const|let|var)\s+', stripped_line):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif any(char in stripped_line for char in ['=', '(', ')', '{', '}', '[', ']']) and not stripped_line.startswith('- '):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif in_potential_code_block and not stripped_line:
            cleaned_lines.append(line)
        elif len(stripped_line) > 0 and not re.match(r'^[a-zA-Z\s,;.:-]*$', stripped_line):
             cleaned_lines.append(line)
             in_potential_code_block = True
        else:
            if in_potential_code_block and stripped_line:
                break 
            pass

    final_code = "\n".join(cleaned_lines).strip()
    final_code = re.sub(r'```(.*?)```', r'\1', final_code, flags=re.DOTALL)
    
    detected_lang = deteksi_bahasa_pemrograman_dari_konten(final_code)
    logger.info(f"{COLOR_YELLOW}[LLM] ⚠ Code extracted with aggressive cleaning. Detected language: {detected_lang}{COLOR_RESET}")
    return final_code.strip(), detected_lang


# === Function: Detect Programming Language from Code Content ===
def deteksi_bahasa_pemrograman_dari_konten(code_content: str) -> str:
    """
    Detects programming language from code content based on heuristics.
    """
    if not code_content:
        return "txt"

    code_content_lower = code_content.lower()

    if "import" in code_content_lower or "def " in code_content_lower or "class " in code_content_lower or ".py" in code_content_lower:
        return "python"
    if "bash" in code_content_lower or "#!/bin/bash" in code_content or "#!/bin/sh" in code_content or "echo " in code_content_lower or ".sh" in code_content_lower:
        return "bash"
    if "function" in code_content_lower or "console.log" in code_content_lower or "const " in code_content_lower or "let " in code_content_lower or "var " in code_content_lower or ".js" in code_content_lower:
        return "javascript"
    if "<html" in code_content_lower or "<body" in code_content_lower or "<div" in code_content_lower:
        return "html"
    if "body {" in code_content_lower or "background-color" in code_content_lower or "color:" in code_content_lower:
        return "css"
    if "<?php" in code_content_lower or (re.search(r'\becho\b', code_content_lower) and not re.search(r'\bbash\b', code_content_lower)):
        return "php"
    if "public class" in code_content_lower or "public static void main" in code_content_lower or ".java" in code_content_lower:
        return "java"
    if "#include <" in code_content_lower or "int main()" in code_content_lower or ".c" in code_content_lower or ".cpp" in code_content_lower:
        return "c"
    return "txt"


# === Function: Detect User Intent ===
def deteksi_niat_pengguna(pesan_pengguna: str) -> str:
    """
    Detects user intent (run shell command, create program, or general conversation).
    Returns string: "shell", "program", or "conversation".
    """
    messages = [
        {"role": "system", "content": """You are an intent detector. Identify whether the user's message intends to:
- "shell": If the user wants to run a system command or perform file operations (e.g., "delete file", "list directory", "run", "open", "install", "compress").
- "program": If the user wants to create or fix code (e.g., "create a python function", "write javascript code", "fix this error", "write a program").
- "conversation": For all other types of questions or interactions that are not direct commands or code generation.

Return only one word from the categories above. Do not provide additional explanations.
"""},
        {"role": "user", "content": f"Detect intent for: '{pesan_pengguna}'"}
    ]
    logger.info(f"{COLOR_BLUE}[AI] Detecting user intent for '{pesan_pengguna}' ({INTENT_DETECTION_MODEL})...{COLOR_RESET}\n")
    
    success, niat = call_llm(messages, INTENT_DETECTION_MODEL, OPENROUTER_API_KEY, max_tokens=10, temperature=0.0)
    
    if success:
        niat_cleaned = niat.strip().lower()
        if not niat_cleaned:
            logger.warning(f"[AI] Empty intent from LLM. Defaulting to 'conversation'.")
            return "conversation"
        elif niat_cleaned in ["shell", "program", "conversation"]:
            return niat_cleaned
        else:
            logger.warning(f"[AI] Unknown intent from LLM: '{niat_cleaned}'. Defaulting to 'conversation'.")
            return "conversation"
    else:
        logger.error(f"[AI] Failed to detect intent: {niat}. Defaulting to 'conversation'.")
        return "conversation"

# === Function: Detect Programming Language requested in Prompt ===
def deteksi_bahasa_dari_prompt(prompt: str) -> str | None:
    """
    Detects the programming language requested in the user's prompt.
    Returns language string (e.g., "python", "bash", "javascript") or None if not specific.
    """
    prompt_lower = prompt.lower()
    if "python" in prompt_lower or "python script" in prompt_lower or "python function" in prompt_lower:
        return "python"
    elif "bash" in prompt_lower or "shell" in prompt_lower or "shell script" in prompt_lower or "sh program" in prompt_lower:
        return "bash"
    elif "javascript" in prompt_lower or "js" in prompt_lower or "nodejs" in prompt_lower:
        return "javascript"
    elif "html" in prompt_lower or "web page" in prompt_lower:
        return "html"
    elif "css" in prompt_lower or "stylesheet" in prompt_lower:
        return "css"
    elif "php" in prompt_lower:
        return "php"
    elif "java" in prompt_lower:
        return "java"
    elif "c++" in prompt_lower or "cpp" in prompt_lower:
        return "cpp"
    elif "c#" in prompt_lower or "csharp" in prompt_lower:
        return "csharp"
    elif "ruby" in prompt_lower or "rb" in prompt_lower:
        return "ruby"
    elif "go lang" in prompt_lower or "golang" in prompt_lower or "go " in prompt_lower:
        return "go"
    elif "swift" in prompt_lower:
        return "swift"
    elif "kotlin" in prompt_lower:
        return "kotlin"
    elif "rust" in prompt_lower:
        return "rust"
    return None


# === Function: Request Code from LLM ===
def minta_kode(prompt: str, error_context: str = None, chat_id: int = None, target_language: str = None) -> tuple[bool, str, str | None]:
    """
    Requests LLM to generate code based on prompt in a specific language.
    If error_context is provided, this is a debugging request.
    Includes recent conversation context if available.
    """
    messages = []
    
    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:]
    for msg in recent_history:
        messages.append(msg)

    # Add system information to the system message
    system_info_message = (
        f"You are an AI coding assistant proficient in various programming languages. "
        f"The system runs on OS: {SYSTEM_INFO['os']}, Shell: {SYSTEM_INFO['shell']}. "
        f"Neofetch output: \n```\n{SYSTEM_INFO['neofetch_output']}\n```\n"
        f"Code results *must* be Markdown code blocks with appropriate language tags "
        f"(e.g., ```python, ```bash, ```javascript, ```html, ```css, ```php, ```java, etc.). "
        f"DO NOT add explanations, intros, conclusions, or extra text outside the Markdown code block. "
        f"Include all necessary imports/dependencies within the code block. "
        f"If any part requires user input, provide clear comments within the code."
    )

    if error_context:
        messages.append({
                "role": "system", 
                "content": system_info_message + " You are fixing code. Based on the error log and provided conversation history, provide *only* the complete fixed code or new code. Ensure the code is directly runnable."
            })
        messages.append({
                "role": "user",
                "content": f"There was an error running the code/command:\n\n{error_context}\n\nFix it or provide complete new code. Focus on {target_language if target_language else 'relevant'} language."
            })
        logger.info(f"{COLOR_BLUE}[AI] Requesting fix/new code ({target_language if target_language else 'universal'}) from AI model ({CODE_GEN_MODEL}) based on error...{COLOR_RESET}\n")
    else:
        messages.append({
                "role": "system", 
                "content": system_info_message
            })
        prompt_with_lang = f"Instruction: {prompt}"
        if target_language:
            prompt_with_lang += f" (in {target_language} language)"
        messages.append({
                "role": "user",
                "content": prompt_with_lang
            })
        logger.info(f"{COLOR_BLUE}[AI] Requesting code ({target_language if target_language else 'universal'}) from AI model ({CODE_GEN_MODEL})...{COLOR_RESET}\n")
    
    success, response_content = call_llm(messages, CODE_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=2048, temperature=0.7)

    if success:
        cleaned_code, detected_language = ekstrak_kode_dari_llm(response_content, target_language)
        return True, cleaned_code, detected_language
    else:
        return False, response_content, None

# === Function: Generate Filename ===
def generate_filename(prompt: str, detected_language: str = "txt") -> str:
    """
    Generates a relevant filename based on user prompt and detected language.
    """
    extension_map = {
        "python": ".py", "bash": ".sh", "javascript": ".js", "html": ".html",
        "css": ".css", "php": ".php", "java": ".java", "c": ".c",
        "cpp": ".cpp", "csharp": ".cs", "ruby": ".rb", "go": ".go",
        "swift": ".swift", "kotlin": ".kt", "rust": ".rs", "txt": ".txt"
    }
    
    messages = [
        {"role": "system", "content": f"You are a filename generator. Provide a single short, relevant, and descriptive filename (no spaces, use underscores, all lowercase, no extension) based on the following code description and language '{detected_language}'. Example: 'factorial_function' or 'cli_calculator'. No explanation, just the filename."},
        {"role": "user", "content": f"Code description: {prompt}"}
    ]
    logger.info(f"{COLOR_BLUE}[AI] Generating filename for '{prompt}' ({FILENAME_GEN_MODEL}) with language {detected_language}...{COLOR_RESET}\n")
    
    success, filename = call_llm(messages, FILENAME_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=20, temperature=0.5)
    
    if not success:
        logger.warning(f"[AI] Failed to generate filename from LLM: {filename}. Using default name.")
        return f"generated_code{extension_map.get(detected_language, '.txt')}"

    filename = filename.strip()
    filename = re.sub(r'[^\w-]', '', filename).lower().replace(' ', '_')
    
    for ext in extension_map.values():
        if filename.endswith(ext):
            filename = filename[:-len(ext)]
            break
            
    if not filename:
        filename = "generated_code"
        
    return filename + extension_map.get(detected_language, '.txt')


# === Function: Convert Natural Language to Shell Command ===
def konversi_ke_perintah_shell(bahasa_natural: str, chat_id: int = None) -> tuple[bool, str]:
    """
    Converts user's natural language into an executable shell command.
    Includes recent conversation context if available.
    """
    messages = []

    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    # Add system information to the system message
    system_message_content = (
        f"You are a natural language to shell command translator. "
        f"The system runs on OS: {SYSTEM_INFO['os']}, Shell: {SYSTEM_INFO['shell']}. "
        f"Convert the following natural language instruction into the most relevant single-line Linux Termux shell command. "
        f"Do not provide explanations, just the command. "
        f"If the instruction is unclear or cannot be converted into a shell command, respond with 'CANNOT_CONVERT'."
    )
    messages.append({"role": "system", "content": system_message_content})
    messages.append({"role": "user", "content": f"Convert this to a shell command: {bahasa_natural}"})

    logger.info(f"{COLOR_BLUE}[AI] Converting natural language to shell command ({COMMAND_CONVERSION_MODEL})...{COLOR_RESET}\n")
    return call_llm(messages, COMMAND_CONVERSION_MODEL, OPENROUTER_API_KEY, max_tokens=128, temperature=0.3)


# === Function: Send Error to LLM for Suggestion ===
def kirim_error_ke_llm_for_suggestion(log_error: str, chat_id: int = None) -> tuple[bool, str]:
    """
    Sends error log to LLM to get suggested fixes.
    Includes recent conversation context if available.
    """
    messages = []

    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    # Add system information to the system message
    system_message_content = (
        f"You are an AI debugger. "
        f"The system runs on OS: {SYSTEM_INFO['os']}, Shell: {SYSTEM_INFO['shell']}. "
        f"Consider this system information when analyzing errors and providing suggestions. "
        f"Provide suggestions in a runnable shell format if possible, or in a Markdown code block. "
        f"Otherwise, provide a brief explanation."
    )

    messages.append({"role": "system", "content": system_message_content})
    messages.append({"role": "user", "content": f"The following error occurred:\n\n{log_error}\n\nWhat is the best suggestion to fix it in a Linux Termux system context? "})
    
    headers = {"HTTP-Referer": "[https://t.me/dseAI_bot](https://t.me/dseAI_bot)"}
    
    logger.info(f"{COLOR_BLUE}[AI] Sending error to AI model ({ERROR_FIX_MODEL}) for suggestions...{COLOR_RESET}\n")
    return call_llm(messages, ERROR_FIX_MODEL, OPENROUTER_API_KEY, max_tokens=512, temperature=0.7, extra_headers=headers)

# === Function: Request General Conversation Answer from LLM ===
def minta_jawaban_konversasi(chat_id: int, prompt: str) -> tuple[bool, str]:
    """
    Requests a general conversational answer from LLM, while maintaining history
    and including references from previous interactions (code, commands).
    """
    history = get_chat_history(chat_id)
    user_context = get_user_context(chat_id)
    
    system_context_messages = []

    # Add system information to the beginning of the system message
    system_context_messages.append(
        {"role": "system", "content": f"System runs on OS: {SYSTEM_INFO['os']}, Shell: {SYSTEM_INFO['shell']}. Neofetch information:\n```\n{SYSTEM_INFO['neofetch_output']}\n```"}
    )

    if user_context["last_command_run"] and user_context["last_ai_response_type"] == "shell":
        system_context_messages.append(
            {"role": "system", "content": f"User just ran a shell command: `{user_context['last_command_run']}`. Consider this context in your answer."}
        )
    if user_context["last_generated_code"] and user_context["last_ai_response_type"] == "program":
        lang_display = user_context["last_generated_code_language"] if user_context["last_generated_code_language"] else "code"
        system_context_messages.append(
            {"role": "system", "content": f"User just received {lang_display} code:\n```{lang_display}\n{user_context['last_generated_code']}\n```. Consider this context in your answer."}
        )
    if user_context["last_error_log"] and user_context["last_user_message_intent"] == "shell":
        system_context_messages.append(
            {"role": "system", "content": f"User encountered an error after running a command: `{user_context['last_command_run']}` with error log:\n```\n{user_context['full_error_output'][-500:]}\n```. Consider this in your answer."}
        )
    elif user_context["last_error_log"] and user_context["last_user_message_intent"] == "program":
         system_context_messages.append(
            {"role": "system", "content": f"User encountered an error after interacting with a program:\n```\n{user_context['full_error_output'][-500:]}\n```. Consider this in your answer."}
        )

    messages_to_send = []
    messages_to_send.extend(system_context_messages)

    max_history_length = 10
    recent_history = history[-max_history_length:]
    messages_to_send.extend(recent_history)
    
    messages_to_send.append({"role": "user", "content": prompt})

    logger.info(f"{COLOR_BLUE}[AI] Requesting conversational answer from AI model ({CONVERSATION_MODEL})...{COLOR_RESET}\n")
    success, response = call_llm(messages_to_send, CONVERSATION_MODEL, OPENROUTER_API_KEY, max_tokens=256, temperature=0.7)

    if success:
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})
        chat_histories[chat_id] = history
    return success, response


# === Function: Save to file ===
def simpan_ke_file(nama_file: str, isi: str) -> bool:
    """
    Saves string content to a file.
    Returns True on success, False on failure.
    """
    try:
        with open(nama_file, "w") as f:
            f.write(isi)
        logger.info(f"{COLOR_GREEN}[FILE] ✅ Code successfully saved to file: {nama_file}{COLOR_RESET}")
        return True
    except IOError as e:
        logger.error(f"[FILE] 🔴 Failed to save file {nama_file}: {e}")
        return False

# === Function: Send Telegram notification ===
async def kirim_ke_telegram(chat_id: int, context: CallbackContext, pesan_raw: str):
    """
    Sends a message to Telegram. Removes ANSI colors and applies MarkdownV2 escaping
    to the entire message content, specifically handling code blocks by not escaping their internal content.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(f"[Telegram] ⚠ Telegram BOT Token or Chat ID not found. Notification not sent.")
        return

    # Remove ANSI color codes first
    pesan_bersih_tanpa_ansi = re.sub(r'\033\[[0-9;]*m', '', pesan_raw)
    
    final_message_parts = []
    
    # 1. Separate by multiline code blocks (```)
    multiline_split = re.split(r'(```(?:\w+)?\n.*?```)', pesan_bersih_tanpa_ansi, flags=re.DOTALL)
    
    for ml_part in multiline_split:
        if ml_part.startswith('```') and ml_part.endswith('```'):
            # This is a multiline code block, add as is (content inside is not escaped)
            final_message_parts.append(ml_part)
        else:
            # This is a text part outside multiline code blocks, now check for inline code blocks (`)
            inline_split = re.split(r'(`[^`]+`)', ml_part) # Capture `...`
            for il_part in inline_split:
                if il_part.startswith('`') and il_part.endswith('`'):
                    # This is an inline code block, add as is (content inside is not escaped)
                    final_message_parts.append(il_part)
                else:
                    # This is plain text, apply MarkdownV2 escaping using the new function
                    final_message_parts.append(_escape_plaintext_markdown_v2(il_part))
            
    pesan_final = "".join(final_message_parts)

    try:
        await context.bot.send_message(chat_id=chat_id, text=pesan_final, parse_mode=ParseMode.MARKDOWN_V2)
        logger.info(f"[Telegram] Notification successfully sent to {chat_id}.")
    except Exception as e:
        logger.error(f"[Telegram] 🔴 Failed to send message to Telegram: {e}")

# === Function: Detect shell commands in AI suggestions ===
def deteksi_perintah_shell(saran_ai: str) -> str | None:
    """
    Detects shell command lines from AI suggestions, including those in
    Markdown code blocks or inline quotes.
    Priority: Markdown code blocks > Inline quotes > Regular Regex patterns
    """
    code_block_pattern = r"```(?:bash|sh|zsh|\w+)?\n(.*?)```"
    inline_code_pattern = r"`([^`]+)`"

    code_blocks = re.findall(code_block_pattern, saran_ai, re.DOTALL)
    for block in code_blocks:
        lines_in_block = [line.strip() for line in block.split('\n') if line.strip()]
        if lines_in_block:
            first_line = lines_in_block[0]
            if any(first_line.startswith(kw) for kw in ["sudo", "apt", "pkg", "pip", "python", "bash", "sh", "./", "chmod", "chown", "mv", "cp", "rmdir", "mkdir", "cd", "ls", "git", "curl", "wget", "tar", "unzip", "zip", "export"]):
                return first_line

    inline_codes = re.findall(inline_code_pattern, saran_ai)
    for code in inline_codes:
        code = code.strip()
        if code and any(code.startswith(kw) for kw in ["sudo", "apt", "pkg", "pip", "python", "bash", "sh", "./", "chmod", "chown", "mv", "cp", "rmdir", "mkdir", "cd", "ls", "git", "curl", "wget", "tar", "unzip", "zip", "export"]):
            return code

    shell_command_patterns = [
        r"^(sudo|apt|pkg|dpkg|pip|python|bash|sh|./|chmod|chown|mv|cp|rmdir|mkdir|cd|ls|grep|find|nano|vi|vim|git|curl|wget|tar|unzip|zip|export|alias)\s+",
        r"^(\S+\.sh)\s+",
        r"^\S+\s+(--\S+|\S+)+",
    ]
    lines = saran_ai.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern in shell_command_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return line
                
    return None

# === Security Function: Filter Dangerous Commands ===
def is_command_dangerous(command: str) -> bool:
    """
    Checks if a shell command contains forbidden keywords.
    """
    command_lower = command.lower()
    
    dangerous_patterns = [
        r'\brm\b\s+-rf',
        r'\brm\b\s+/\s*',
        r'\bpkg\s+uninstall\b',
        r'\bmv\b\s+/\s*',
        r'\bchown\b\s+root',
        r'\bchmod\b\s+\d{3}\s+/\s*',
        r'\bsu\b',
        r'\bsudo\b\s+poweroff',
        r'\breboot\b',
        r'\bformat\b',
        r'\bmkfs\b',
        r'\bdd\b',
        r'\bfdisk\b',
        r'\bparted\b',
        r'\bwipefs\b',
        r'\bcrontab\b\s+-r',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            logger.warning(f"[SECURITY] ❗ Dangerous command detected: {command}")
            return True
    return False

# === Mode Function: Shell Observation and Error Correction (for Telegram) ===
async def run_shell_observer_telegram(command_to_run: str, update: Update, context: CallbackContext):
    """
    Runs a shell command, monitors output, and sends logs/error suggestions to Telegram.
    Non-interactive.
    """
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    user_context["last_command_run"] = command_to_run
    user_context["full_error_output"] = []
    
    telegram_log_buffer = [] 
    async def send_telegram_chunk():
        nonlocal telegram_log_buffer
        if telegram_log_buffer:
            # This is formatted as a code block with 'log' tag
            escaped_log_content = "\n".join(telegram_log_buffer)
            # Since _escape_plaintext_markdown_v2 doesn't escape *, we need to manually
            # add asterisks for formatting here if desired.
            # However, for logs, leave it as is and Telegram will display it as a code block.
            message = f"```log\n{escaped_log_content}\n```" 
            await kirim_ke_telegram(chat_id, context, message)
            telegram_log_buffer = []

    # Initial message when running the command. Command_to_run is inserted directly into backticks.
    await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Starting command: `{command_to_run}`")
    logger.info(f"\n{COLOR_BLUE}[Shell] 🟢 Running command: `{command_to_run}`{COLOR_RESET}\n")

    # shlex.quote is only used to safely run the command in the shell, not for Markdown display.
    safe_command_to_run = shlex.quote(command_to_run)

    try:
        child = pexpect.spawn(f"bash -c {safe_command_to_run}", encoding='utf-8', timeout=None)
    except pexpect.exceptions.ExceptionPexpect as e:
        error_msg = f"*❗ SHELL ERROR* Failed to run command: `{str(e)}`. Ensure the command is valid, bash is available, and pexpect is installed correctly."
        await kirim_ke_telegram(chat_id, context, error_msg)
        logger.error(f"[Shell] 🔴 Failed to run command: {e}")
        return ConversationHandler.END

    error_detected_in_stream = False
    error_line_buffer = []
    user_context["last_error_log"] = None

    while True:
        try:
            line = await asyncio.to_thread(child.readline)

            if not line:
                if child.eof():
                    logger.info(f"{COLOR_GREEN}[Shell] ✅ Shell process finished.{COLOR_RESET}")
                    await send_telegram_chunk()
                    await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Shell command finished.")
                    
                    if user_context["last_error_log"]:
                        await kirim_ke_telegram(chat_id, context, f"*❗ ERROR* Error detected in last execution. Do you want to debug this program with AI assistance? (Yes/No)")
                        user_context["awaiting_debug_response"] = True
                        return DEBUGGING_STATE
                    break
                continue

            cleaned_line = line.strip()
            logger.info(f"{COLOR_YELLOW}[Shell Log] {cleaned_line}{COLOR_RESET}")
            
            telegram_log_buffer.append(cleaned_line)
            if len(telegram_log_buffer) >= 10:
                await send_telegram_chunk()

            error_line_buffer.append(cleaned_line)
            if len(error_line_buffer) > 10:
                error_line_buffer.pop(0)
            
            user_context["full_error_output"].append(cleaned_line)

            is_program_execution_command = bool(re.match(r"^(python|sh|bash|node|\./)\s+\S+\.(py|sh|js|rb|pl|php)", command_to_run, re.IGNORECASE))
            
            if is_program_execution_command and any(keyword in cleaned_line.lower() for keyword in ["error", "exception", "not found", "failed", "permission denied", "command not found", "no such file or directory", "segmentation fault", "fatal"]):
                if not error_detected_in_stream:
                    error_detected_in_stream = True
                    await send_telegram_chunk()
                    
                    user_context["last_error_log"] = "\n".join(user_context["full_error_output"])
                    
                    await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Error detected. Requesting AI suggestions...")
                    logger.info(f"{COLOR_RED}[AI] Error detected. Sending context to model...{COLOR_RESET}\n")
                    
                    success_saran, saran = kirim_error_ke_llm_for_suggestion(user_context["last_error_log"], chat_id)

                    # Detect language from suggestion for proper syntax highlighting
                    saran_lang = deteksi_bahasa_pemrograman_dari_konten(saran) if success_saran else "text"

                    if success_saran:
                        telegram_msg = f"""*❗ ERROR DETECTED*
*Latest Error Log:*
```log
{user_context["full_error_output"][-2000:]}
```

---

*💡 AI SUGGESTION*
```{saran_lang}
{saran}
```
"""
                    else:
                        telegram_msg = f"*❗ ERROR DETECTED*\n*Latest Error Log:*\n```log\n{user_context['full_error_output'][-2000:]}\n```\n\n*🔴 AI ERROR* Failed to get AI suggestion: {saran}"
                    
                    await kirim_ke_telegram(chat_id, context, telegram_msg)

        except pexpect.exceptions.EOF:
            logger.info(f"{COLOR_GREEN}[Shell] ✅ Shell process finished.{COLOR_RESET}")
            await send_telegram_chunk()
            await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Shell command finished.")
            if user_context["last_error_log"]:
                await kirim_ke_telegram(chat_id, context, f"*❗ ERROR* Error detected in last execution. Do you want to debug this program with AI assistance? (Yes/No)")
                user_context["awaiting_debug_response"] = True
                return DEBUGGING_STATE
            break
        except KeyboardInterrupt:
            logger.warning(f"\n{COLOR_YELLOW}[Shell] ✋ Interrupted by Termux user.{COLOR_RESET}")
            child.sendline('\x03')
            child.close()
            await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Shell process manually stopped.")
            break
        except Exception as e:
            error_msg = f"*🔴 INTERNAL ERROR* An unexpected error occurred in `shell_observer`: `{str(e)}`"
            await kirim_ke_telegram(chat_id, context, error_msg)
            logger.error(f"[Shell] 🔴 Unexpected error: {e}")
            if child.isalive():
                child.close()
            break
    
    return ConversationHandler.END

# === Function to check system info with neofetch ===
def check_system_info():
    """
    Checks for neofetch availability and retrieves system information.
    If neofetch is not available, it attempts to install it based on the OS.
    """
    global SYSTEM_INFO

    def install_neofetch(install_command):
        """Attempts to install neofetch using the given command."""
        logger.info(f"{COLOR_YELLOW}[INFO SISTEM] Neofetch tidak ditemukan. Mencoba menginstal dengan: '{' '.join(install_command)}'...{COLOR_RESET}")
        try:
            # Try to install silently
            subprocess.run(install_command, check=True, capture_output=True)
            logger.info(f"{COLOR_GREEN}[INFO SISTEM] Neofetch berhasil diinstal.{COLOR_RESET}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"{COLOR_RED}🔴 ERROR: Gagal menginstal neofetch. Output error:\n{e.stderr.strip()}{COLOR_RESET}")
            return False
        except FileNotFoundError:
            logger.error(f"{COLOR_RED}🔴 ERROR: Perintah instalasi '{install_command[0]}' tidak ditemukan.{COLOR_RESET}")
            return False

    # First, try to detect the OS
    detected_os = "Unknown"
    try:
        if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
            detected_os = "Termux"
        elif os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                os_release_content = f.read()
                if "ID=debian" in os_release_content or "ID=ubuntu" in os_release_content:
                    detected_os = "Debian/Ubuntu"
                elif "ID_LIKE=arch" in os_release_content:
                    detected_os = "Arch Linux"
                elif "ID=fedora" in os_release_content:
                    detected_os = "Fedora"
                # Add more OS detection as needed
    except Exception as e:
        logger.warning(f"[INFO SISTEM] Gagal mendeteksi OS secara detail: {e}. Menggunakan deteksi default.")
    
    SYSTEM_INFO["os"] = detected_os
    logger.info(f"[INFO SISTEM] OS yang terdeteksi: {SYSTEM_INFO['os']}")

    # Check if neofetch is installed
    try:
        subprocess.run(["which", "neofetch"], check=True, capture_output=True)
        neofetch_installed = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        neofetch_installed = False

    if not neofetch_installed:
        if detected_os == "Termux":
            success_install = install_neofetch(["pkg", "install", "-y", "neofetch"])
        elif detected_os == "Debian/Ubuntu":
            success_install = install_neofetch(["sudo", "apt", "install", "-y", "neofetch"])
            if not success_install: # Fallback if sudo isn't configured for current user
                 success_install = install_neofetch(["apt", "install", "-y", "neofetch"])
        elif detected_os == "Arch Linux":
            success_install = install_neofetch(["sudo", "pacman", "-S", "--noconfirm", "neofetch"])
        elif detected_os == "Fedora":
            success_install = install_neofetch(["sudo", "dnf", "install", "-y", "neofetch"])
        else:
            logger.warning(f"{COLOR_YELLOW}[INFO SISTEM] Neofetch tidak ditemukan dan OS tidak dikenal untuk instalasi otomatis. Coba instal neofetch secara manual.{COLOR_RESET}")
            success_install = False
        
        if not success_install:
            logger.error(f"{COLOR_RED}🔴 ERROR: Neofetch tidak terinstal dan gagal menginstalnya secara otomatis. Tidak dapat melanjutkan tanpa neofetch.{COLOR_RESET}")
            exit(1) # Exit if installation fails

    try:
        # Run neofetch and capture its output
        result = subprocess.run(["neofetch", "--off", "--config", "none", "--stdout"], capture_output=True, text=True, check=True)
        neofetch_output = result.stdout.strip()
        SYSTEM_INFO["neofetch_output"] = neofetch_output
        logger.info(f"{COLOR_GREEN}[INFO SISTEM] Neofetch successfully executed.{COLOR_RESET}")

        # Parse neofetch output to get OS and Shell
        os_match = re.search(r"OS:\s*(.*?)\n", neofetch_output)
        shell_match = re.search(r"Shell:\s*(.*?)\n", neofetch_output)

        if os_match:
            SYSTEM_INFO["os"] = os_match.group(1).strip()
        if shell_match:
            SYSTEM_INFO["shell"] = shell_match.group(1).strip()
        
        logger.info(f"[INFO SISTEM] Detected OS: {SYSTEM_INFO['os']}")
        logger.info(f"[INFO SISTEM] Detected Shell: {SYSTEM_INFO['shell']}")

    except subprocess.CalledProcessError as e:
        logger.error(f"{COLOR_RED}🔴 ERROR: Failed to run neofetch even after installation attempt. Error output:\n{e.stderr.strip()}{COLOR_RESET}")
        logger.error(f"{COLOR_RED}Please check neofetch installation and your PATH manually.{COLOR_RESET}")
        logger.error(f"{COLOR_RED}Program will stop. Please fix neofetch installation to continue.{COLOR_RESET}")
        exit(1)
    except FileNotFoundError:
        # This case should ideally not happen if install_neofetch worked
        logger.error(f"{COLOR_RED}🔴 ERROR: 'neofetch' command not found after installation attempt.{COLOR_RESET}")
        logger.error(f"{COLOR_RED}Program will stop. Please fix neofetch installation to continue.{COLOR_RESET}")
        exit(1)
    except Exception as e:
        logger.error(f"{COLOR_RED}🔴 ERROR: An unexpected error occurred while checking system information: {e}{COLOR_RESET}")
        logger.error(f"{COLOR_RED}Program will stop.{COLOR_RESET}")
        exit(1)

# === Telegram Command Handlers ===

async def start_command(update: Update, context: CallbackContext):
    """Sends a welcome message when the /start command is given."""
    chat_id = update.effective_chat.id
    pesan_raw = f"""
*Halo! Saya AI Asisten Shell & Kode Anda.*

Saya bisa membantu Anda dengan beberapa hal:

* *⚙️ SHELL*
    Jalankan perintah sistem atau operasi file. Cukup ketik perintah Anda atau instruksi alami (misal: `tampilkan isi direktori`).
* *✨ PROGRAM*
    Hasilkan atau perbaiki kode program. Cukup berikan instruksi kode (misal: `buatkan fungsi python untuk menghitung faktorial`, `bikinin program bash simple berfungsi sebagai kalkulator`, `tulis kode javascript untuk DOM`). Saya akan mendeteksi bahasanya.
* *💬 KONVERSASI*
    Ajukan pertanyaan umum atau mulai percakapan santai.

---

*Informasi Sistem Terdeteksi:*
* *OS:* {SYSTEM_INFO['os']}
* *Shell:* {SYSTEM_INFO['shell']}

---

*Perintah Tambahan:*
* `/listfiles` - Melihat daftar file yang dihasilkan.
* `/deletefile <nama_file>` - Menghapus file yang dihasilkan.
* `/clear_chat` - Menghapus riwayat percakapan.

*Penting:* Pastikan bot saya berjalan di Termux dan semua variabel lingkungan sudah diatur!
    """
    await kirim_ke_telegram(chat_id, context, pesan_raw)
    logger.info(f"[Telegram] /start message sent to {chat_id}.")

async def handle_listfiles_command(update: Update, context: CallbackContext):
    """Handles the /listfiles command to display a list of generated files."""
    chat_id = update.effective_chat.id

    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ ACCESS DENIED* You are not authorized to use this feature. Contact the bot admin.")
        logger.warning(f"[Auth] ⚠ Unauthorized access attempt /listfiles from {chat_id}.")
        return

    allowed_extensions = [
        '.py', '.sh', '.js', '.html', '.css', '.php', '.java', '.c', '.cpp',
        '.cs', '.rb', '.go', '.swift', '.kt', '.rs', '.txt'
    ]
    
    files = [f for f in os.listdir('.') if os.path.isfile(f) and any(f.endswith(ext) for ext in allowed_extensions) and f != os.path.basename(__file__)]
    
    if files:
        # Use backticks for filenames to look like inline code
        file_list_msg = "*📄 MY FILES* List of available program files:\n" + "\n".join([f"- `{f}`" for f in files])
    else:
        file_list_msg = "*📄 MY FILES* No program files generated by the bot."
    
    await kirim_ke_telegram(chat_id, context, file_list_msg)
    logger.info(f"[Telegram] File list sent to {chat_id}.")

async def handle_deletefile_command(update: Update, context: CallbackContext):
    """Handles the /deletefile command to delete a specific file."""
    chat_id = update.effective_chat.id
    filename_to_delete = " ".join(context.args).strip()

    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ ACCESS DENIED* You are not authorized to use this feature. Contact the bot admin.")
        logger.warning(f"[Auth] ⚠ Unauthorized access attempt /deletefile from {chat_id}.")
        return

    if not filename_to_delete:
        await kirim_ke_telegram(chat_id, context, f"*❓ COMMAND* Please provide the filename to delete. Example: `/deletefile your_program_name.py`")
        return

    allowed_extensions = [
        '.py', '.sh', '.js', '.html', '.css', '.php', '.java', '.c', '.cpp',
        '.cs', '.rb', '.go', '.swift', '.kt', '.rs', '.txt'
    ]
    is_allowed_extension = any(filename_to_delete.endswith(ext) for ext in allowed_extensions)

    if not is_allowed_extension or filename_to_delete == os.path.basename(__file__):
        await kirim_ke_telegram(chat_id, context, f"*❗ DENIED* Only generated program files can be deleted. You cannot delete the bot's own file or files with disallowed extensions.")
        logger.warning(f"[Security] ❗ Attempt to delete invalid file: {filename_to_delete} from {chat_id}")
        return

    try:
        if os.path.exists(filename_to_delete) and os.path.isfile(filename_to_delete):
            os.remove(filename_to_delete)
            await kirim_ke_telegram(chat_id, context, f"*✅ SUCCESS* File `{filename_to_delete}` successfully deleted.")
            logger.info(f"[File] File {filename_to_delete} deleted by {chat_id}.")
        else:
            await kirim_ke_telegram(chat_id, context, f"*❗ NOT FOUND* File `{filename_to_delete}` not found.")
    except Exception as e:
        await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR* Failed to delete file `{filename_to_delete}`: `{str(e)}`")
        logger.error(f"[File] 🔴 Failed to delete file {filename_to_delete}: {e}")

async def handle_clear_chat_command(update: Update, context: CallbackContext):
    """Handles the /clear_chat command to clear conversation history."""
    chat_id = update.effective_chat.id
    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ ACCESS DENIED* You are not authorized to use this feature. Contact the bot admin.")
        logger.warning(f"[Auth] ⚠ Unauthorized access attempt /clear_chat from {chat_id}.")
        return

    if chat_id in chat_histories:
        del chat_histories[chat_id]
        await kirim_ke_telegram(chat_id, context, f"*✅ SUCCESS* Your conversation history has been cleared.")
        logger.info(f"[Chat] Chat history for {chat_id} cleared.")
    else:
        await kirim_ke_telegram(chat_id, context, f"*💬 INFO* No conversation history to clear.")


async def handle_text_message(update: Update, context: CallbackContext):
    """
    Handles all non-command text messages from Telegram.
    Will detect user intent and call appropriate functions.
    """
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()
    user_context = get_user_context(chat_id)
    
    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ ACCESS DENIED* You are not authorized to interact with this bot. Contact the bot admin.")
        logger.warning(f"[Auth] ⚠ Unauthorized access attempt from {chat_id}: {user_message}")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    niat = deteksi_niat_pengguna(user_message)
    user_context["last_user_message_intent"] = niat
    logger.info(f"[Intent] User {chat_id} -> Intent: {niat}")

    if niat == "shell":
        await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Intent detected: Shell Command. Translating instruction: `{user_message}`")
        success_konversi, perintah_shell = konversi_ke_perintah_shell(user_message, chat_id)
        perintah_shell = perintah_shell.strip()

        if not success_konversi:
            await kirim_ke_telegram(chat_id, context, f"*🔴 CONVERSION ERROR* An issue occurred while converting the command:\n```\n{perintah_shell}\n```")
            logger.error(f"[Error] Conversion Failed: {perintah_shell}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        elif perintah_shell == "CANNOT_CONVERT":
            await kirim_ke_telegram(chat_id, context, f"*❗ UNCLEAR COMMAND* Sorry, I cannot convert that instruction into a clear shell command. Please provide more specific instructions.")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return

        if is_command_dangerous(perintah_shell):
            await kirim_ke_telegram(chat_id, context, f"*🚫 PROHIBITED* This command is not allowed to be executed: `{perintah_shell}`. Please use another command.")
            logger.warning(f"[Security] ❗ Attempt to run dangerous command: {perintah_shell} from {chat_id}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        
        await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Translated shell command: `{perintah_shell}`")
        user_context["last_ai_response_type"] = "shell"
        user_context["last_command_run"] = perintah_shell
        user_context["last_generated_code"] = None
        user_context["last_generated_code_language"] = None
        # Call run_shell_observer_telegram and handle return value for ConversationHandler
        return await run_shell_observer_telegram(perintah_shell, update, context)

    elif niat == "program":
        await kirim_ke_telegram(chat_id, context, f"*✨ PROGRAM* Intent detected: Program Creation. Starting code generation for: `{user_message}`")
        
        target_lang_from_prompt = deteksi_bahasa_dari_prompt(user_message)

        success_code, kode_tergenerasi, detected_language = minta_kode(user_message, chat_id=chat_id, target_language=target_lang_from_prompt)

        if not success_code:
            await kirim_ke_telegram(chat_id, context, f"*🔴 CODE GENERATION ERROR* An issue occurred while generating code:\n```\n{kode_tergenerasi}\n```")
            logger.error(f"[Error] Code Gen Failed: {kode_tergenerasi}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        
        generated_file_name = generate_filename(user_message, detected_language)
        simpan_ok = simpan_ke_file(generated_file_name, kode_tergenerasi)

        if simpan_ok:
            user_context["last_generated_code"] = kode_tergenerasi
            user_context["last_generated_code_language"] = detected_language
            user_context["last_ai_response_type"] = "program"
            user_context["last_command_run"] = None

            await kirim_ke_telegram(chat_id, context, f"*✅ SUCCESS* *{detected_language.capitalize()}* code successfully generated and saved to `{generated_file_name}`.")
            await kirim_ke_telegram(chat_id, context, f"*You can open it in Termux with:* `nano {generated_file_name}`")
            
            run_command_suggestion = ""
            if detected_language == "python":
                run_command_suggestion = f"`python {generated_file_name}`"
            elif detected_language == "bash":
                run_command_suggestion = f"`bash {generated_file_name}` or `chmod +x {generated_file_name} && ./{generated_file_name}`"
            elif detected_language == "javascript":
                run_command_suggestion = f"`node {generated_file_name}` (ensure Node.js is installed)"
            elif detected_language == "html":
                run_command_suggestion = f"Open this file in your web browser."
            elif detected_language == "php":
                run_command_suggestion = f"`php {generated_file_name}` (ensure PHP is installed)"
            elif detected_language == "java":
                run_command_suggestion = f"Compile with `javac {generated_file_name}` then run with `java {generated_file_name.replace('.java', '')}`"
            elif detected_language in ["c", "cpp"]:
                run_command_suggestion = f"Compile with `gcc {generated_file_name} -o a.out` then run with `./a.out`"
            
            if run_command_suggestion:
                await kirim_ke_telegram(chat_id, context, f"*And run with:* {run_command_suggestion}")

            await kirim_ke_telegram(chat_id, context, f"*📋 GENERATED CODE*\n```{detected_language}\n{kode_tergenerasi}\n```")
        else:
            await kirim_ke_telegram(chat_id, context, f"*🔴 FILE ERROR* Failed to save generated code to file.")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
        return ConversationHandler.END
            

    else: # niat == "conversation"
        await kirim_ke_telegram(chat_id, context, f"*💬 GENERAL CONVERSATION* Intent detected: General Conversation. Requesting AI answer...")
        success_response, jawaban_llm = minta_jawaban_konversasi(chat_id, user_message)
        user_context["last_ai_response_type"] = "conversation"
        user_context["last_command_run"] = None
        user_context["last_generated_code"] = None
        user_context["last_generated_code_language"] = None
        
        if not success_response:
            await kirim_ke_telegram(chat_id, context, f"*🔴 AI ERROR* An issue occurred while processing the conversation:\n```\n{jawaban_llm}\n```")
            logger.error(f"[Error] Conversation Failed: {jawaban_llm}")
        else:
            await kirim_ke_telegram(chat_id, context, f"*💬 AI RESPONSE*\n{jawaban_llm}")
        return ConversationHandler.END


async def handle_unknown_command(update: Update, context: CallbackContext):
    """Responds to unknown commands (e.g., /foo bar)."""
    chat_id = update.effective_chat.id
    await kirim_ke_telegram(chat_id, context, f"*❓ UNKNOWN* Unknown command. Please use `/start` to see available commands.")
    logger.warning(f"[Command] ⚠ Unknown command from {chat_id}: {update.message.text}")

# === Debugging Conversation Handler ===
async def ask_for_debug_response(update: Update, context: CallbackContext):
    """Asks for a Yes/No response from the user for debugging."""
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    
    if user_context["awaiting_debug_response"]:
        user_response = update.message.text.strip().lower()
        if user_response in ["ya", "yes"]:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Starting debugging session...")
            logger.info(f"{COLOR_BLUE}[Debug] Starting debugging for {chat_id}{COLOR_RESET}")
            
            error_log = user_context["last_error_log"]
            last_command = user_context["last_command_run"]
            last_generated_code_lang = user_context["last_generated_code_language"]

            if error_log:
                await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Requesting LLM to analyze error and provide fix/new code...")
                success_debug, debug_saran, debug_lang = minta_kode(prompt="", error_context=error_log, chat_id=chat_id, target_language=last_generated_code_lang)
                
                if not success_debug:
                    await kirim_ke_telegram(chat_id, context, f"*🔴 DEBUGGING ERROR* An issue occurred during debugging:\n```\n{debug_saran}\n```")
                    logger.error(f"[Debug] Debug Failed: {debug_saran}")
                else:
                    # Try to extract filename from the last executed command
                    debug_file_name = None
                    if last_command:
                        match = re.search(r"^(python|sh|bash|node|php|\./)\s+(\S+\.(py|sh|js|rb|pl|php|java|c|cpp|html|css|txt))", last_command, re.IGNORECASE)
                        if match:
                            debug_file_name = match.group(2)
                    
                    if not debug_file_name:
                        # If unable to extract from command, create a new filename
                        debug_file_name = generate_filename("bug_fix", debug_lang)


                    simpan_ok = simpan_ke_file(debug_file_name, debug_saran)
                    if simpan_ok:
                        user_context["last_generated_code"] = debug_saran
                        user_context["last_generated_code_language"] = debug_lang
                        user_context["last_ai_response_type"] = "program"
                        await kirim_ke_telegram(chat_id, context, f"*✅ SUCCESS* AI has generated a fix/new code to `{debug_file_name}`.")
                        
                        run_command_suggestion = ""
                        if debug_lang == "python":
                            run_command_suggestion = f"`python {debug_file_name}`"
                        elif debug_lang == "bash":
                            run_command_suggestion = f"`bash {debug_file_name}` or `chmod +x {debug_file_name} && ./{debug_file_name}`"
                        elif debug_lang == "javascript":
                            run_command_suggestion = f"`node {debug_file_name}` (ensure Node.js is installed)"
                        elif debug_lang == "html":
                            run_command_suggestion = f"Open this file in your web browser."
                        elif debug_lang == "php":
                            run_command_suggestion = f"`php {debug_file_name}` (ensure PHP is installed)"
                        elif debug_lang == "java":
                            run_command_suggestion = f"Compile with `javac {debug_file_name}` then run with `java {debug_file_name.replace('.java', '')}`"
                        elif debug_lang in ["c", "cpp"]:
                            run_command_suggestion = f"Compile with `gcc {debug_file_name} -o a.out` then run with `./a.out`"

                        if run_command_suggestion:
                            await kirim_ke_telegram(chat_id, context, f"*Please review and try running again with:* {run_command_suggestion}\n\n*📋 FIX CODE*\n```{debug_lang}\n{debug_saran}\n```")
                        else:
                            await kirim_ke_telegram(chat_id, context, f"*Please review and try running again. *\n\n*📋 FIX CODE*\n```{debug_lang}\n{debug_saran}\n```")

                    else:
                        await kirim_ke_telegram(chat_id, context, f"*🔴 FILE ERROR* Failed to save generated fix code to file.")
                        user_context["last_generated_code"] = None
                        user_context["last_ai_response_type"] = None
                        user_context["last_generated_code_language"] = None
            else:
                await kirim_ke_telegram(chat_id, context, f"*💬 INFO* No error log available for debugging.")
            
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END
        elif user_response in ["tidak", "no"]:
            await kirim_ke_telegram(chat_id, context, f"*💬 INFO* Debugging canceled.")
            logger.info(f"{COLOR_GREEN}[Debug] Debugging canceled by {chat_id}{COLOR_RESET}")
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END
        else:
            await kirim_ke_telegram(chat_id, context, f"*❓ INVALID RESPONSE* Please answer 'Yes' or 'No'.")
            return DEBUGGING_STATE
    else:
        # If not in debugging state, proceed to regular handle_text_message
        return await handle_text_message(update, context)


def main():
    """Main function to start the Telegram bot."""
    # Call function to check system information at startup
    check_system_info()

    if not TELEGRAM_BOT_TOKEN:
        logger.error(f"ERROR: TELEGRAM_BOT_TOKEN is not set. Please set the environment variable or enter it directly.")
        return
    if not TELEGRAM_CHAT_ID:
        logger.error(f"ERROR: TELEGRAM_CHAT_ID is not set. Please set the environment variable or enter it directly.")
        return
    if not OPENROUTER_API_KEY:
        logger.error(f"ERROR: OPENROUTER_API_KEY is not set. Please set the environment variable or enter it directly.")
        return

    logger.info(f"{COLOR_GREEN}Starting Telegram Bot...{COLOR_RESET}")
    logger.info(f"Using TOKEN: {'*' * (len(TELEGRAM_BOT_TOKEN) - 5) + TELEGRAM_BOT_TOKEN[-5:] if len(TELEGRAM_BOT_TOKEN) > 5 else TELEGRAM_BOT_TOKEN}")
    logger.info(f"Allowed Chat ID: {TELEGRAM_CHAT_ID}")

    # Build Application with JobQueue without explicit tzinfo in its constructor
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).job_queue(JobQueue()).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("listfiles", handle_listfiles_command))
    application.add_handler(CommandHandler("deletefile", handle_deletefile_command))
    application.add_handler(CommandHandler("clear_chat", handle_clear_chat_command))
    
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_debug_response)],
        states={
            DEBUGGING_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_debug_response)],
        },
        fallbacks=[CommandHandler('cancel', lambda update, context: ConversationHandler.END)]
    )
    application.add_handler(conv_handler)
    
    # Text message handler must be placed after ConversationHandler so ConversationHandler has priority
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    application.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    logger.info(f"{COLOR_GREEN}Bot is running. Press Ctrl+C to stop.{COLOR_RESET}")
    # Use run_polling directly, as it manages its own event loop
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info(f"{COLOR_GREEN}Bot stopped.{COLOR_RESET}")

if __name__ == "__main__":
    main()
