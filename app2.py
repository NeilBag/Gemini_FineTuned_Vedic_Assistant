import os
import logging
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
# Note: We are NOT importing Flask-Session anymore
from werkzeug.utils import secure_filename
from docx import Document
try:
    from xhtml2pdf import pisa
    from PIL import Image
except ImportError:
    pisa = None
    Image = None
from dotenv import load_dotenv # For loading local .env file
import google.generativeai as genai
import tempfile # For export files
import traceback # For error logging
import json # For reading/writing history files
import uuid # For generating unique session IDs
import html # For escaping text for PDF export

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# --- Logging Configuration ---
# Configure logging to be compatible with Render (stdout/stderr)
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(), # Allow setting log level via env var
    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
    # Force logging to stdout/stderr for Render compatibility by default
    # If logs still don't show, explicitly add handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configuration Class ---
class Configuration:
    def __init__(self):
        # General Flask settings
        self.DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

        # SECRET_KEY is crucial for signing the standard Flask session cookie (which holds session_id)
        self.SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
        if not self.SECRET_KEY:
            logger.critical("CRITICAL: FLASK_SECRET_KEY not set! Using insecure default. SET THIS IN PRODUCTION.")
            self.SECRET_KEY = 'change-this-in-production-to-a-very-secure-random-string' # Insecure fallback

        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB Max upload size

        # --- Paths for Render Persistent Disks ---
        # Path for user file uploads
        render_upload_disk_path = os.getenv('RENDER_DISK_PATH_UPLOADS')
        self.UPLOAD_FOLDER = render_upload_disk_path if render_upload_disk_path else 'uploads'
        logger.info(f"Upload folder configured to: {self.UPLOAD_FOLDER}")

        # Path for chat history JSON files
        render_history_disk_path = os.getenv('RENDER_DISK_PATH_CHAT_HISTORIES')
        self.CHAT_HISTORY_FOLDER = render_history_disk_path if render_history_disk_path else './chat_histories'
        logger.info(f"Chat history folder configured to: {self.CHAT_HISTORY_FOLDER}")
        # --- End Persistent Disk Paths ---

        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'} # Allowed upload types

        # Standard Flask Session Cookie settings (for the session_id)
        self.SESSION_COOKIE_SECURE = os.getenv('FLASK_ENV', 'development') == 'production' # Use secure cookies in production
        self.SESSION_COOKIE_HTTPONLY = True
        self.SESSION_COOKIE_SAMESITE = 'Lax' # Recommended for most cases

# --- Generative AI Settings Class ---
class GenerativeAISettings:
    def __init__(self):
        self.API_KEY = os.getenv("GEMINI_API_KEY")
        # Allow overriding model name via environment variable
        self.MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "tunedModels/hinduism-veda-expert-v1") # Your fine-tuned model
        self.GENERATION_CONFIG = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 128000, # Adjust based on model limits
            "response_mime_type": "text/plain",
        }

# --- App Creation Factory Function ---
def create_app(config=Configuration(), ai_settings=GenerativeAISettings()):
    app = Flask(__name__)
    app.config.from_object(config) # Load config from Configuration object

    # --- Ensure Upload and History Directories Exist ---
    # Critical for Render persistent disks
    for dir_path in [app.config['UPLOAD_FOLDER'], app.config['CHAT_HISTORY_FOLDER']]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory verified/created: {dir_path}")
        except OSError as e:
            logger.critical(f"FATAL: Could not create required directory {dir_path}: {e}")
            # Stop app startup if essential directories can't be created
            raise RuntimeError(f"Failed to create directory: {dir_path}, error: {e}") from e

    # --- Initialize Generative AI ---
    app.ai_model = None # Initialize attribute
    if not ai_settings.API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set. AI features will be disabled.")
    else:
        try:
            genai.configure(api_key=ai_settings.API_KEY)
            app.ai_model = genai.GenerativeModel(
                model_name=ai_settings.MODEL_NAME,
                generation_config=ai_settings.GENERATION_CONFIG
            )
            logger.info(f"Generative AI model '{ai_settings.MODEL_NAME}' initialized successfully.")
        except Exception as e:
            logger.exception(f"CRITICAL: Failed to configure or initialize Generative AI model '{ai_settings.MODEL_NAME}'. AI features disabled. Error: {e}")
            # Keep app.ai_model as None

    return app

# --- Create the Flask App Instance ---
app = create_app()
# Convenience reference to the model (might be None if init failed)
model = getattr(app, 'ai_model', None)


# --- Chat History File Helper Functions ---
def get_history_filepath(session_id):
    """Constructs the full path for a given session's history file."""
    if not session_id or not isinstance(session_id, str):
        return None
    # Basic sanitization for filename safety
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ('-', '_'))
    if not safe_session_id or len(safe_session_id) < 10: # Avoid overly short/empty IDs
        logger.error(f"Invalid session_id for file path: {session_id}")
        return None
    filename = f"{safe_session_id}.json"
    return os.path.join(app.config['CHAT_HISTORY_FOLDER'], filename)

def load_history(session_id):
    """Loads chat history from a JSON file. Returns empty list on failure."""
    filepath = get_history_filepath(session_id)
    if not filepath:
        logger.warning("Attempted to load history with invalid session_id.")
        return []
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            if isinstance(history_data, list):
                logger.info(f"Loaded history for session {session_id} from {filepath} ({len(history_data)} entries)")
                return history_data
            else:
                logger.error(f"Invalid data format in history file {filepath}. Expected list, got {type(history_data)}. Discarding.")
                # Consider renaming/deleting corrupted file: os.remove(filepath)
                return []
        else:
            # This is normal for a new session
            logger.info(f"No history file found for session {session_id} at {filepath}. Starting fresh.")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from history file {filepath}: {e}. Discarding.")
        # Consider renaming/deleting corrupted file: os.remove(filepath)
        return []
    except Exception as e:
        logger.exception(f"Error loading history file {filepath}: {e}")
        return [] # Return empty list on other errors

def save_history(session_id, history_data):
    """Saves chat history list to a JSON file. Returns True on success."""
    filepath = get_history_filepath(session_id)
    if not filepath:
         logger.error("Attempted to save history with invalid session_id.")
         return False
    if not isinstance(history_data, list):
         logger.error(f"Attempted to save non-list data ({type(history_data)}) as history for session {session_id}.")
         return False
    try:
        # Ensure the directory exists right before writing
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2) # Use indent for readability during debug
        logger.info(f"Saved history for session {session_id} to {filepath} ({len(history_data)} entries)")
        return True
    except Exception as e:
        logger.exception(f"Error saving history file {filepath}: {e}")
        return False

def delete_history_file(session_id):
    """Deletes the history file for a given session. Returns True if gone or deleted."""
    filepath = get_history_filepath(session_id)
    if not filepath:
        logger.error("Attempted to delete history with invalid session_id.")
        return False
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted history file for session {session_id}: {filepath}")
            return True
        else:
            logger.info(f"Attempted to delete non-existent history file for session {session_id}: {filepath}")
            return True # Success if already gone
    except Exception as e:
        logger.exception(f"Error deleting history file {filepath}: {e}")
        return False


# --- File Handling Helper Functions ---
def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def secure_save_file(file):
    """Saves the file securely to the UPLOAD_FOLDER. Returns (filepath, original_filename) or (None, None)."""
    if not file or not file.filename:
        logger.warning("secure_save_file called with invalid file object.")
        return None, None

    original_filename = file.filename # Keep original name for reference
    filename = secure_filename(original_filename)
    if not filename: # Handle cases like filenames being just '..'
        filename = f"unnamed_upload_{uuid.uuid4().hex[:8]}" # Generate unique name
        logger.warning(f"Original filename '{original_filename}' was invalid, saved as '{filename}'")

    # Use the configured upload folder path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        # Ensure directory exists right before saving (belt-and-suspenders)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        logger.info(f"File '{original_filename}' saved successfully to: {file_path}")
        return file_path, original_filename
    except Exception as e:
        logger.exception(f"Failed to save file '{filename}' (from '{original_filename}') to '{file_path}': {e}")
        return None, None

def read_file_content(file_path):
    """Reads content from supported text-based files. Returns string content or error message."""
    if not file_path or not os.path.exists(file_path):
         logger.error(f"File not found for reading content: {file_path}")
         return "(Error: Uploaded file could not be found on server)"

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    content = ""
    logger.info(f"Reading content from: {file_path} (extension: {extension})")
    try:
        if extension == '.txt':
            # Try common encodings
            for encoding in ['utf-8', 'latin-1', 'windows-1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"Read .txt file '{os.path.basename(file_path)}' with encoding {encoding}")
                    break
                except UnicodeDecodeError: continue
                except Exception as read_e: raise read_e
            else:
                 logger.warning(f"Could not decode .txt file {os.path.basename(file_path)} with common encodings.")
                 content = f"(System note: Could not automatically decode .txt file: {os.path.basename(file_path)})"

        elif extension == '.docx':
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            logger.info(f"Read .docx file '{os.path.basename(file_path)}' content.")
        elif extension in ['.png', '.jpg', '.jpeg'] and Image:
             try:
                 with Image.open(file_path) as img:
                    img.verify()
                    content = f"(System note: Image file detected: {os.path.basename(file_path)}, format: {img.format}, size: {img.size})"
                 logger.info(f"Processed image file metadata for '{os.path.basename(file_path)}'.")
             except Exception as img_e:
                 logger.warning(f"Could not process image file {os.path.basename(file_path)}: {img_e}")
                 content = f"(System note: Could not fully process image file: {os.path.basename(file_path)})"
        elif extension == '.pdf':
             content = f"(System note: PDF file detected: {os.path.basename(file_path)} - content extraction not implemented)"
             logger.info(f"PDF file '{os.path.basename(file_path)}' detected (extraction not implemented).")
        else:
            logger.warning(f"Content reading not implemented for file extension: {extension} ({os.path.basename(file_path)})")
            return f"(System note: File type {extension} not processed for content: {os.path.basename(file_path)})"

        # Limit content size
        max_chars = 15000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [System note: Content truncated from file {os.path.basename(file_path)}]"
            logger.warning(f"Truncated content from {file_path} to {max_chars} characters.")
        return content

    except Exception as e:
        logger.exception(f"Error reading file content from {file_path}: {e}")
        return f"(System note: Error reading file {os.path.basename(file_path)})"


# --- History Processing Helper Functions ---
def clean_and_validate_history_for_api(history_list):
    """Cleans a history list (from file) to ensure alternating roles and non-empty messages."""
    if not isinstance(history_list, list):
        logger.error(f"Invalid history_list type: {type(history_list)}. Returning empty list.")
        return []
    if not history_list: return []

    logger.info(f"Cleaning history with {len(history_list)} raw entries for API call.")
    cleaned_api_history = []
    expected_role = "user"

    for i, entry in enumerate(history_list):
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dict history entry at index {i}: {entry}")
            continue

        user_msg = entry.get('user')
        model_msg = entry.get('model')
        has_user_text = isinstance(user_msg, str) and user_msg.strip()
        has_model_text = isinstance(model_msg, str) and model_msg.strip()

        added_role = None
        if has_user_text and expected_role == "user":
            cleaned_api_history.append({'role': 'user', 'parts': [{"text": user_msg.strip()}]})
            added_role = "user"
            expected_role = "model"
        elif has_model_text and expected_role == "model":
             cleaned_api_history.append({'role': 'model', 'parts': [{"text": model_msg.strip()}]})
             added_role = "model"
             expected_role = "user"
        # Handle cases where entry only has one role matching expectation (less common with file storage)
        elif has_user_text and not has_model_text and expected_role == "user":
             cleaned_api_history.append({'role': 'user', 'parts': [{"text": user_msg.strip()}]})
             added_role = "user"
             expected_role = "model"
        elif has_model_text and not has_user_text and expected_role == "model":
             cleaned_api_history.append({'role': 'model', 'parts': [{"text": model_msg.strip()}]})
             added_role = "model"
             expected_role = "user"
        else:
            reason = "empty content"
            if expected_role == "user" and has_model_text: reason = "out of order (expected user, got model)"
            if expected_role == "model" and has_user_text: reason = "out of order (expected model, got user)"
            if not has_user_text and not has_model_text: reason = "both parts empty"
            logger.warning(f"Skipping history entry at index {i} during API prep: Reason: {reason}. Entry: {entry}. Expected role: {expected_role}")

    logger.info(f"Cleaned history for API contains {len(cleaned_api_history)} entries.")
    return cleaned_api_history

def build_history_for_template(history_list):
    """Builds history suitable for rendering in the template."""
    template_history = []
    if not isinstance(history_list, list): return []
    for entry in history_list:
        if isinstance(entry, dict):
             template_history.append({
                 'user': entry.get('user'), # Pass None if missing
                 'model': entry.get('model') # Pass None if missing
             })
    return template_history


# --- Export Helper Functions ---
def export_docx(text):
    """Exports text to a DOCX file using a temporary file. Returns filepath or None."""
    if not isinstance(text, str): text = str(text)
    try:
        document = Document()
        document.add_paragraph(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", mode='wb') as temp_file:
            docx_path = temp_file.name
            document.save(docx_path)
        logger.info(f"Generated DOCX for export: {docx_path}")
        return docx_path
    except Exception as e:
        logger.exception(f"Error creating DOCX file: {e}")
        return None

def export_pdf(text):
    """Exports text to a PDF file using a temporary file. Returns filepath or None."""
    if pisa is None:
        logger.error("xhtml2pdf library is not available. Cannot export PDF.")
        return None
    if not isinstance(text, str): text = str(text)
    escaped_text = html.escape(text) # Basic HTML escaping

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as temp_file:
            pdf_path = temp_file.name
            html_content = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Export</title></head><body><pre>{escaped_text}</pre></body></html>"
            from io import BytesIO
            pdf_buffer = BytesIO()
            pisa_status = pisa.CreatePDF(BytesIO(html_content.encode('UTF-8')), dest=pdf_buffer, encoding='UTF-8')
            if pisa_status.err:
                logger.error(f"Error creating PDF using pisa: {pisa_status.err}")
                return None
            temp_file.write(pdf_buffer.getvalue())
        logger.info(f"Generated PDF for export: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.exception(f"Error creating PDF file: {e}")
        return None


# --- Main Route ---
@app.route("/", methods=["GET", "POST"])
def index2():
    # --- Session ID Management ---
    if 'session_id' not in session:
        session['session_id'] = uuid.uuid4().hex
        logger.info(f"Generated new session ID: {session['session_id']}")
    session_id = session['session_id']
    # --- End Session ID Management ---

    current_model = getattr(app, 'ai_model', None)
    # Load history fresh from file for every request involving it
    current_chat_history = load_history(session_id)

    # --- POST Request Handling ---
    if request.method == "POST":
        # Check if AI is needed and available
        action = request.form.get("action", "submit")
        if action == "submit" and not current_model:
             flash("AI Model is not available. Cannot process prompt. Please check server status.", "danger")
             logger.error("Attempted to process prompt but AI model is not available.")
             return redirect(url_for('index2'))

        try:
            prompt = request.form.get("prompt", "").strip()
            export_format = request.form.get("export_format", "")

            # --- File Upload Processing ---
            uploaded_file_contents = []
            uploaded_filenames = []
            files = request.files.getlist("files")
            if files:
                logger.info(f"Received {len(files)} file(s) for upload.")
                for file in files:
                    if file and file.filename: # Check if file object seems valid
                        if allowed_file(file.filename):
                            file_path, original_filename = secure_save_file(file)
                            if file_path:
                                content = read_file_content(file_path)
                                uploaded_file_contents.append(f"--- Content from {original_filename} ---\n{content}\n--- End of {original_filename} ---")
                                uploaded_filenames.append(original_filename)
                            else: flash(f"Could not save allowed file: {secure_filename(file.filename)}", "warning")
                        else: flash(f"File type not allowed: {secure_filename(file.filename)}", "warning")
                    elif file: logger.warning("Received a file input with no filename.")

            # --- Handle 'Submit' Action ---
            if action == "submit":
                full_prompt = prompt
                if uploaded_file_contents:
                    full_prompt = "\n\n".join(uploaded_file_contents) + "\n\n---\nUser Prompt:\n" + prompt

                if not prompt and not uploaded_file_contents:
                    flash("Please enter a prompt or upload a file.", "warning")
                    return redirect(url_for('index2'))

                # Clean history list specifically for the API call
                api_history = clean_and_validate_history_for_api(current_chat_history)

                # Log data before API call (DEBUG level)
                logger.debug(f"Session {session_id}: Attempting AI call. Prompt (start): {full_prompt[:300]}...")
                try: logger.debug(f"Session {session_id}: API History ({len(api_history)} entries): {json.dumps(api_history, indent=1)}")
                except Exception as log_e: logger.error(f"Session {session_id}: Error logging API history: {log_e}")

                response_text = None
                try:
                    logger.info(f"Session {session_id}: Starting chat session and sending message to model '{app.ai_model.model_name}'.")
                    chat_session = current_model.start_chat(history=api_history)
                    response = chat_session.send_message(full_prompt)
                    response_text = response.text # Extract text response
                    logger.info(f"Session {session_id}: Received response (length: {len(response_text)}).")
                except Exception as ai_error:
                     logger.error(f"Session {session_id}: Error during Generative AI call: {ai_error}", exc_info=True)
                     flash(f"Error communicating with AI. Please check server logs. ({type(ai_error).__name__})", "danger")
                     return redirect(url_for('index2')) # Redirect on failure

                # Update history list and save to file
                history_user_entry = prompt # Store user's typed prompt
                if uploaded_filenames:
                    history_user_entry += f"\n\n[System note: Used uploaded file(s): {', '.join(uploaded_filenames)}]"

                if response_text and response_text.strip():
                    current_chat_history.append({'user': history_user_entry, 'model': response_text})
                    save_history(session_id, current_chat_history) # Save updated list
                else:
                    logger.warning(f"Session {session_id}: AI returned an empty response. Not saving this turn to history.")
                    flash("AI returned an empty response.", "info")

                return redirect(url_for('index2')) # PRG Pattern

            # --- Handle 'Export' Action ---
            elif action == "export":
                if not export_format:
                    flash("Please select an export format.", "warning")
                    return redirect(url_for('index2'))

                last_model_response = None
                # Iterate backwards through the current history list
                for entry in reversed(current_chat_history):
                    if isinstance(entry, dict) and 'model' in entry and isinstance(entry['model'], str) and entry['model'].strip():
                        last_model_response = entry['model']
                        break

                if not last_model_response:
                    flash("No valid AI response found in history to export.", "warning")
                    return redirect(url_for('index2'))

                logger.info(f"Session {session_id}: Exporting last AI response as {export_format}.")
                export_path, mimetype, download_name = None, None, None

                if export_format == "docx":
                    export_path = export_docx(last_model_response)
                    mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    download_name="ai_response.docx"
                elif export_format == "pdf":
                    if pisa:
                        export_path = export_pdf(last_model_response)
                        mimetype='application/pdf'
                        download_name="ai_response.pdf"
                    else:
                        flash("PDF export library unavailable.", "danger")
                        return redirect(url_for('index2'))

                if export_path and mimetype and download_name:
                    try:
                        logger.info(f"Session {session_id}: Sending export file: {export_path}")
                        return send_file(export_path, mimetype=mimetype, as_attachment=True, download_name=download_name)
                    finally: # Cleanup attempt after file is sent (or if send fails)
                        try:
                            if export_path and os.path.exists(export_path): os.remove(export_path)
                            logger.info(f"Session {session_id}: Cleaned up temp export file: {export_path}")
                        except OSError as e: logger.error(f"Session {session_id}: Error removing temp export file {export_path}: {e}")
                else:
                    flash(f"Failed to generate export file for {export_format}. Check logs.", "danger")
                    return redirect(url_for('index2'))

            else: # Unknown action
                 logger.warning(f"Session {session_id}: Received unknown form action: {action}")
                 flash("Invalid form action.", "warning")
                 return redirect(url_for('index2'))

        except Exception as e: # Catch-all for POST processing
            logger.error(f"Session {session_id}: Unhandled error processing POST request: {str(e)}", exc_info=True)
            flash(f"An unexpected server error occurred processing your request.", "danger")
            return redirect(url_for('index2'))

    # --- GET Request Handling ---
    template_history = build_history_for_template(current_chat_history)
    ai_model_status = "available" if current_model else "unavailable"
    return render_template("index2.html", history=template_history, ai_model_status=ai_model_status)


# --- Clear History Route ---
@app.route("/clear", methods=["POST"])
def clear_history():
    session_id = session.get('session_id')
    if session_id:
        deleted = delete_history_file(session_id)
        if deleted: flash("Chat history cleared.", "info")
        else: flash("Could not clear chat history file.", "warning")
        # Keep the session_id cookie so user continues with the same ID but empty history
    else:
        flash("No active session found.", "info")
    return redirect(url_for('index2'))


# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 Not Found error for URL: {request.url}")
    # Optional: Create a templates/404.html page
    return "<h2>404 Not Found</h2><p>The page you requested could not be found.</p><a href='/'>Go Home</a>", 404

@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    logger.warning(f"Request entity too large (limit: {max_size_mb:.0f}MB). URL: {request.url}, Client: {request.remote_addr}")
    flash(f"Uploaded file(s) are too large. Maximum allowed total size is {max_size_mb:.0f}MB.", "danger")
    return redirect(url_for('index2'))

@app.errorhandler(500)
def internal_error(error):
    # Log the detailed error. exc_info=True includes the traceback.
    logger.error(f"Internal Server Error (500): {error}", exc_info=True)
    # Avoid flashing detailed errors to the user in production
    flash("An internal server error occurred. We have been notified and are looking into it.", "danger")
    return redirect(url_for('index2')) # Redirect to a clean page


# --- Main Execution Guard (for local development) ---
if __name__ == "__main__":
    logger.info("Starting Flask development server...")
    # Check if essential local dev config is missing
    if not app.config.get('SECRET_KEY') or app.config.get('SECRET_KEY') == 'change-this-in-production-to-a-very-secure-random-string':
        logger.critical("SECURITY WARNING: Running locally with insecure default SECRET_KEY. Set FLASK_SECRET_KEY environment variable.")
    if not getattr(app, 'ai_model', None):
         logger.warning("WARNING: Running locally but AI Model failed to initialize. Check GEMINI_API_KEY environment variable and logs.")

    # Use host='0.0.0.0' to be accessible on your local network
    # Port defaults to 5000 if PORT env var isn't set
    # Debug mode is controlled by FLASK_DEBUG env var via Configuration class
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=app.config['DEBUG'])
