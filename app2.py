import os
import logging
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from flask_session import Session # Import Session
from werkzeug.utils import secure_filename
from docx import Document
# Consider potential issues with xhtml2pdf dependencies, ensure installed correctly
try:
    from xhtml2pdf import pisa
    # Pillow is often needed implicitly by xhtml2pdf or other image libs
    from PIL import Image
except ImportError:
    pisa = None
    Image = None # Define Image as None if Pillow isn't available or needed directly here
from dotenv import load_dotenv # For loading API key from .env file locally
import google.generativeai as genai
import tempfile # For unique temporary files for export
import traceback # For detailed error logging
import json # For pretty printing logs

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# Configure logging
# In production (like Render), you might want to configure logging differently,
# e.g., logging to stdout/stderr which Render captures.
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(), # Allow setting log level via env var
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # Force logging to stdout/stderr for Render compatibility
    # handlers=[logging.StreamHandler()] # Uncomment this line if logs aren't showing up in Render
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Configuration:
    def __init__(self):
        # General Flask settings
        self.DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
        # IMPORTANT: Set FLASK_SECRET_KEY in Render Environment Variables
        self.SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
        if not self.SECRET_KEY:
            logger.warning("FLASK_SECRET_KEY not set! Using insecure default for local dev.")
            self.SECRET_KEY = 'a_very_secret_key_for_local_dev_only' # Fallback ONLY for local dev

        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

        # --- Paths for Render Persistent Disks (with local fallbacks) ---
        # Assumes disks named 'flask-sessions' and 'uploads' are mounted at
        # /mnt/data/flask_session and /mnt/data/uploads respectively in Render.
        # Render sets env vars like RENDER_DISK_PATH_<disk_name> automatically.
        # Use these paths if running on Render, otherwise use local relative paths.

        # Path for Flask-Session files
        # Use '/var/data/' prefix if Render adds it, otherwise use '/mnt/data/' convention
        render_session_disk_path = os.getenv('RENDER_DISK_PATH_FLASK_SESSION', None)
        self.SESSION_FILE_DIR = render_session_disk_path if render_session_disk_path else './.flask_session'
        logger.info(f"Session file directory configured to: {self.SESSION_FILE_DIR}")


        # Path for user file uploads
        render_upload_disk_path = os.getenv('RENDER_DISK_PATH_UPLOADS', None)
        self.UPLOAD_FOLDER = render_upload_disk_path if render_upload_disk_path else 'uploads'
        logger.info(f"Upload folder configured to: {self.UPLOAD_FOLDER}")
        # --- End Persistent Disk Paths ---

        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'} # Allowed upload types

        # Flask-Session specific config
        self.SESSION_TYPE = 'filesystem' # Use filesystem for sessions
        self.SESSION_PERMANENT = False # Session expires when browser closes (can be configured)
        self.SESSION_USE_SIGNER = True # Sign the session cookie identifier (needs SECRET_KEY)
        self.SESSION_COOKIE_SECURE = os.getenv('FLASK_ENV', 'development') == 'production' # Use secure cookies in production
        self.SESSION_COOKIE_HTTPONLY = True
        self.SESSION_COOKIE_SAMESITE = 'Lax' # Or 'Strict'


# --- Generative AI Settings ---
class GenerativeAISettings:
    def __init__(self):
        # IMPORTANT: Set GEMINI_API_KEY in Render Environment Variables
        self.API_KEY = os.getenv("GEMINI_API_KEY")
        self.MODEL_NAME = "tunedModels/hinduism-veda-expert-v1" # Or your fine-tuned model name if applicable
        self.GENERATION_CONFIG = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 128000,
            "response_mime_type": "text/plain",
        }

# --- App Creation ---
def create_app(config=Configuration(), ai_settings=GenerativeAISettings()):
    app = Flask(__name__)
    app.config.from_object(config)

    # --- Ensure Session and Upload Directories Exist ---
    # These directories MUST exist before initializing Session or handling uploads.
    # Use the paths defined in the config (which check for Render env vars).
    try:
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        logger.info(f"Session directory verified/created: {app.config['SESSION_FILE_DIR']}")
    except OSError as e:
        logger.error(f"FATAL: Could not create session directory {app.config['SESSION_FILE_DIR']}: {e}")
        # Depending on severity, might want to raise an exception to stop app startup
        raise RuntimeError(f"Failed to create session directory: {e}") from e

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info(f"Upload directory verified/created: {app.config['UPLOAD_FOLDER']}")
    except OSError as e:
        logger.error(f"FATAL: Could not create upload directory {app.config['UPLOAD_FOLDER']}: {e}")
        raise RuntimeError(f"Failed to create upload directory: {e}") from e

    # --- Initialize Flask-Session ---
    # Initialize AFTER config is loaded and session directory exists
    Session(app)

    # --- Initialize Generative AI ---
    app.ai_model = None # Initialize as None
    if not ai_settings.API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set. AI features disabled.")
        # App can continue running, but AI calls will fail gracefully later
    else:
        try:
            genai.configure(api_key=ai_settings.API_KEY)
            app.ai_model = genai.GenerativeModel(
                model_name=ai_settings.MODEL_NAME,
                generation_config=ai_settings.GENERATION_CONFIG
            )
            logger.info(f"Generative AI model '{ai_settings.MODEL_NAME}' initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to configure or initialize Generative AI: {e}. AI features disabled.")
            # Ensure model remains None if initialization fails

    return app

app = create_app()
# Convenience reference to the model stored in app context (might be None)
model = getattr(app, 'ai_model', None)

# --- File Handling ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def secure_save_file(file):
    """Saves the file securely and returns its path and original filename."""
    if not file or not file.filename:
        logger.warning("secure_save_file called with invalid file object.")
        return None, None
    filename = secure_filename(file.filename)
    if not filename:
        filename = f"unnamed_upload_{os.urandom(4).hex()}" # Generate a unique name for safety
        logger.warning(f"Original filename was invalid, saved as {filename}")

    # Use the configured upload folder path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        logger.info(f"File '{file.filename}' saved successfully to: {file_path}")
        return file_path, file.filename # Return original filename for user display
    except Exception as e:
        logger.exception(f"Failed to save file '{filename}' to '{file_path}': {e}")
        return None, None

def read_file_content(file_path):
    """Reads content from supported text-based files."""
    if not file_path or not os.path.exists(file_path):
         logger.error(f"File not found for reading content: {file_path}")
         return "(Error: Uploaded file not found on server)"

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
                    logger.info(f"Read .txt file with encoding {encoding}")
                    break # Stop if successful
                except UnicodeDecodeError:
                    continue # Try next encoding
                except Exception as read_e: # Catch other read errors
                    raise read_e # Re-raise other errors
            else: # If loop completes without break
                 logger.warning(f"Could not decode .txt file {file_path} with common encodings.")
                 content = f"(Could not decode .txt file: {os.path.basename(file_path)})"

        elif extension == '.docx':
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs if para.text])
            logger.info("Read .docx file content.")
        # Basic image handling example (just confirms it's an image)
        elif extension in ['.png', '.jpg', '.jpeg'] and Image:
             try:
                 with Image.open(file_path) as img: # Use 'with' to ensure file handle is closed
                    img.verify() # Verify image data without fully loading pixels
                    content = f"(Image file detected: {os.path.basename(file_path)}, format: {img.format}, size: {img.size})"
                 logger.info("Processed image file metadata.")
             except Exception as img_e:
                 logger.warning(f"Could not process image file {file_path}: {img_e}")
                 content = f"(Could not fully process image file: {os.path.basename(file_path)})"
        elif extension == '.pdf':
             # Placeholder: Add PDF text extraction here if needed (e.g., using PyMuPDF)
             # import fitz # Requires PyMuPDF
             # try:
             #     doc = fitz.open(file_path)
             #     content = "".join(page.get_text() for page in doc).strip()
             #     doc.close()
             #     logger.info("Read .pdf file content.")
             # except Exception as pdf_e:
             #     logger.error(f"Error reading PDF {file_path}: {pdf_e}")
             #     content = f"(Error reading PDF file: {os.path.basename(file_path)})"
             content = f"(PDF file detected: {os.path.basename(file_path)} - content extraction not implemented)"
             logger.info("PDF file detected (extraction not implemented).")
        else:
            logger.warning(f"Content reading not implemented for extension: {extension}")
            return f"(File type {extension} not processed for content)"

        # Limit content size to avoid overly large prompts
        max_chars = 15000 # Increased limit slightly, adjust as needed per model context window
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated from original file {os.path.basename(file_path)}]"
            logger.warning(f"Truncated content from {file_path} to {max_chars} characters.")
        return content

    except Exception as e:
        logger.exception(f"Error reading file content from {file_path}: {e}")
        return f"(Error reading file {os.path.basename(file_path)})"


# --- History Handling (with cleaning and validation) ---
def clean_and_validate_session_history(session_history):
    """
    Cleans the session history to ensure alternating roles and non-empty messages.
    Returns a cleaned list suitable for build_history_for_api.
    """
    if not isinstance(session_history, list):
        logger.error(f"Invalid session_history type: {type(session_history)}. Returning empty list.")
        return []
    if not session_history:
        return []

    logger.info(f"Cleaning history with {len(session_history)} raw entries.")
    cleaned_history = []
    expected_role = "user" # History must start with user

    for i, entry in enumerate(session_history):
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dict history entry at index {i}: {entry}")
            continue

        user_msg = entry.get('user')
        model_msg = entry.get('model')

        # Standardize: Check if text exists and is not just whitespace
        has_user_text = isinstance(user_msg, str) and user_msg.strip()
        has_model_text = isinstance(model_msg, str) and model_msg.strip()

        added_role = None
        # Add user message if valid and expected
        if has_user_text and expected_role == "user":
            cleaned_history.append({'role': 'user', 'text': user_msg.strip()})
            added_role = "user"
            expected_role = "model"
        # Add model message if valid and expected
        elif has_model_text and expected_role == "model":
             cleaned_history.append({'role': 'model', 'text': model_msg.strip()})
             added_role = "model"
             expected_role = "user"
        # Handle cases where an entry might *only* have one valid role that matches expectation
        elif has_user_text and not has_model_text and expected_role == "user":
             cleaned_history.append({'role': 'user', 'text': user_msg.strip()})
             added_role = "user"
             expected_role = "model"
        elif has_model_text and not has_user_text and expected_role == "model":
             cleaned_history.append({'role': 'model', 'text': model_msg.strip()})
             added_role = "model"
             expected_role = "user"
        else:
            # Log skipped entry and reason
            reason = "empty content"
            if expected_role == "user" and has_model_text: reason = "out of order (expected user, got model)"
            if expected_role == "model" and has_user_text: reason = "out of order (expected model, got user)"
            if not has_user_text and not has_model_text: reason = "both parts empty"

            logger.warning(f"Skipping history entry at index {i}: Reason: {reason}. Entry: {entry}. Expected role: {expected_role}")
            # Don't change expected_role if we skip, wait for a valid entry of the expected type.

    logger.info(f"Cleaned history contains {len(cleaned_history)} entries.")
    return cleaned_history


def build_history_for_api(cleaned_session_history):
    """Builds history in the format required by the Google Generative AI API
       using the pre-cleaned/validated list."""
    api_history = []
    for entry in cleaned_session_history:
        # Already validated that 'text' is not empty/whitespace in cleaning step
        api_history.append({
            "role": entry['role'],
            "parts": [{"text": entry['text']}]
        })
    return api_history


def build_history_for_template(session_history):
    """Builds history suitable for rendering in the template (no changes needed here)."""
    template_history = []
    if not isinstance(session_history, list): # Add type check for safety
        return []
    for entry in session_history:
         # Ensure we only add dicts with expected keys (even if None)
        if isinstance(entry, dict):
             template_history.append({
                 'user': entry.get('user'),
                 'model': entry.get('model')
             })
    return template_history


# --- Export Functions ---
def export_docx(text):
    """Exports text to a DOCX file using a temporary file."""
    if not isinstance(text, str): text = str(text) # Ensure text is string
    try:
        document = Document()
        # Handle potential invalid XML characters if needed, though usually paragraph handles it
        document.add_paragraph(text)
        # Use tempfile for unique, auto-cleaned file in system temp dir
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", mode='wb') as temp_file:
            docx_path = temp_file.name
            document.save(docx_path) # python-docx saves directly to path
        logger.info(f"Generated DOCX for export: {docx_path}")
        return docx_path
    except Exception as e:
        logger.exception(f"Error creating DOCX file: {e}")
        return None

def export_pdf(text):
    """Exports text to a PDF file using a temporary file."""
    if pisa is None:
        logger.error("xhtml2pdf library is not available. Cannot export PDF.")
        return None
    if not isinstance(text, str): text = str(text) # Ensure text is string

    # Basic HTML escaping for safety, especially if text contains HTML-like chars
    import html
    escaped_text = html.escape(text)

    try:
        # Use tempfile for unique, auto-cleaned file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as temp_file:
            pdf_path = temp_file.name
            # Simple HTML wrapping, <pre> preserves whitespace and line breaks
            html_content = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Export</title></head><body><pre>{escaped_text}</pre></body></html>"

            # Use BytesIO buffer for pisa
            from io import BytesIO
            pdf_buffer = BytesIO()
            # Ensure pisa uses UTF-8
            pisa_status = pisa.CreatePDF(BytesIO(html_content.encode('UTF-8')), dest=pdf_buffer, encoding='UTF-8')

            if pisa_status.err:
                logger.error(f"Error creating PDF using pisa: {pisa_status.err}")
                # Try to log more details from pisa if possible
                return None

            # Write buffer content to the temp file
            temp_file.write(pdf_buffer.getvalue()) # Write directly

        logger.info(f"Generated PDF for export: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.exception(f"Error creating PDF file: {e}")
        return None


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index2():
    # Initialize chat history in session if not present
    if 'chat_history' not in session or not isinstance(session['chat_history'], list):
        logger.info("Initializing new chat history in session.")
        session['chat_history'] = []

    # Check if AI model is available (might be None if init failed)
    current_model = getattr(app, 'ai_model', None)

    if request.method == "POST":
        # Ensure model is available before processing POST actions that need it
        if current_model is None and request.form.get("action", "submit") == "submit":
             flash("AI Model is not available. Cannot process prompt. Please check server status.", "danger")
             return redirect(url_for('index2'))

        try:
            action = request.form.get("action", "submit") # Distinguish submit vs export clicks
            prompt = request.form.get("prompt", "").strip()
            export_format = request.form.get("export_format", "")

            # --- File Upload Processing ---
            uploaded_file_contents = []
            uploaded_filenames = []
            files = request.files.getlist("files")
            if files:
                logger.info(f"Received {len(files)} file(s) for upload.")
                for file in files:
                    # Check if file object seems valid and has a filename
                    if file and file.filename:
                        if allowed_file(file.filename):
                            file_path, original_filename = secure_save_file(file)
                            if file_path:
                                logger.info(f"Processing uploaded file: {original_filename} (path: {file_path})")
                                content = read_file_content(file_path)
                                uploaded_file_contents.append(f"--- Content from {original_filename} ---\n{content}\n--- End of {original_filename} ---")
                                uploaded_filenames.append(original_filename)
                                # Note: We are not deleting uploaded files automatically here.
                                # Consider adding cleanup logic if needed (e.g., based on session expiry or a background job)
                            else:
                                flash(f"Could not save allowed file: {secure_filename(file.filename)}", "warning")
                        else:
                            flash(f"File type not allowed: {secure_filename(file.filename)}", "warning")
                    elif file and not file.filename:
                         logger.warning("Received a file input with no filename.")
                    # else: file object itself might be invalid/empty

            # --- Handle Prompt Submission ---
            if action == "submit":
                # --- Combine Prompt and File Content ---
                full_prompt = prompt
                if uploaded_file_contents:
                    # Prepend file contents to the user's prompt text
                    full_prompt = "\n\n".join(uploaded_file_contents) + "\n\n---\nUser Prompt:\n" + prompt
                    logger.info(f"Combined prompt includes content from files: {', '.join(uploaded_filenames)}")

                if not prompt and not uploaded_file_contents:
                    flash("Please enter a prompt or upload a file.", "warning")
                    # Avoid redirect loop, render directly or redirect to GET
                    return redirect(url_for('index2'))

                # --- Clean History and Call Generative AI ---
                logger.info("Cleaning session history for API call...")
                current_session_history = session.get('chat_history', [])
                cleaned_history_list = clean_and_validate_session_history(current_session_history)
                api_history = build_history_for_api(cleaned_history_list) # Build from cleaned list

                # **** ADD LOGGING ****
                logger.debug(f"Attempting AI call with prompt (first 500 chars): {full_prompt[:500]}...")
                try:
                    # Log history structure being sent (use json for readability)
                    logger.debug(f"API History being sent ({len(api_history)} entries): {json.dumps(api_history, indent=2)}")
                except Exception as log_e:
                    logger.error(f"Error logging API history: {log_e}") # Catch errors during logging itself

                try:
                    logger.info("Starting chat session and sending message...")
                    if not current_model: # Double check model exists before calling methods
                         raise RuntimeError("AI Model is not available.")

                    chat_session = current_model.start_chat(history=api_history)
                    # TODO: Consider adding request_options like timeout if needed
                    response = chat_session.send_message(full_prompt)
                    response_text = response.text
                    logger.info(f"Received response from Generative AI (length: {len(response_text)}).")
                except Exception as ai_error:
                     # Log the specific error from the AI call
                     logger.error(f"Error during Generative AI call: {ai_error}", exc_info=True)
                     # Check for specific API error types if the library provides them
                     flash(f"Error communicating with AI. Please check logs. ({type(ai_error).__name__})", "danger")
                     return redirect(url_for('index2'))


                # --- Update Session History ---
                history_user_entry = prompt # Store only the user's typed prompt for display clarity
                if uploaded_filenames:
                    # Add a note about files used, but don't include content in history display
                    history_user_entry += f"\n\n[Info: Used uploaded file(s): {', '.join(uploaded_filenames)}]"

                # Ensure we don't add empty responses
                if response_text and response_text.strip():
                    session['chat_history'].append({'user': history_user_entry, 'model': response_text})
                    session.modified = True # Mark session as modified
                    logger.info("Appended user prompt and AI response to session history.")
                else:
                    logger.warning("AI returned an empty response. Not added to history.")
                    flash("AI returned an empty response.", "info")


                # Use Post/Redirect/Get pattern to prevent form resubmission on refresh
                return redirect(url_for('index2'))

            # --- Handle Export Request ---
            elif action == "export":
                if not export_format:
                    flash("Please select an export format (DOCX or PDF).", "warning")
                    return redirect(url_for('index2'))

                # Get the text of the *last* model response from history
                last_model_response = None
                current_session_history = session.get('chat_history', [])
                if current_session_history:
                    # Iterate backwards through the raw history to find the last entry with a 'model' key
                    for entry in reversed(current_session_history):
                        if isinstance(entry, dict) and 'model' in entry and isinstance(entry['model'], str) and entry['model'].strip():
                            last_model_response = entry['model']
                            break

                if not last_model_response:
                    flash("No valid AI response found in history to export.", "warning")
                    return redirect(url_for('index2'))

                logger.info(f"Exporting last AI response as {export_format}.")
                export_path = None
                mimetype = None
                download_name = None

                if export_format == "docx":
                    export_path = export_docx(last_model_response)
                    mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    download_name="ai_response.docx"
                elif export_format == "pdf":
                    if pisa: # Only attempt if pisa is available
                        export_path = export_pdf(last_model_response)
                        mimetype='application/pdf'
                        download_name="ai_response.pdf"
                    else:
                        flash("PDF export library (xhtml2pdf) is unavailable.", "danger")
                        return redirect(url_for('index2'))

                if export_path and mimetype and download_name:
                    try:
                        # Use send_file. Cleanup happens after request via 'finally'.
                        logger.info(f"Sending file: {export_path}, mimetype: {mimetype}")
                        return send_file(export_path, mimetype=mimetype, as_attachment=True, download_name=download_name)
                    finally:
                        # Attempt to clean up the temporary file after sending
                        try:
                            if export_path and os.path.exists(export_path):
                                os.remove(export_path)
                                logger.info(f"Cleaned up temporary export file: {export_path}")
                        except OSError as e:
                            # Log error but don't crash the request if cleanup fails
                            logger.error(f"Error removing temporary export file {export_path}: {e}")
                else:
                    # Handle case where export_path wasn't generated
                    flash(f"Failed to generate export file for format {export_format}. Check logs.", "danger")
                    return redirect(url_for('index2'))

            else:
                 # Unknown action
                 logger.warning(f"Received unknown form action: {action}")
                 flash("Invalid form action.", "warning")
                 return redirect(url_for('index2'))


        except Exception as e:
            # General error handler for the POST request processing
            logger.error(f"Unhandled error processing POST request: {str(e)}", exc_info=True) # Use exc_info=True
            flash(f"An unexpected server error occurred. Please try again later.", "danger")
            return redirect(url_for('index2'))

    # --- Handle GET Request ---
    # Build history for rendering the template
    template_history = build_history_for_template(session.get('chat_history', []))
    # Check for AI model status to potentially display a warning banner in template (optional)
    ai_model_status = "available" if current_model else "unavailable"
    return render_template("index2.html", history=template_history, ai_model_status=ai_model_status)


# --- Clear History Route ---
@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop('chat_history', None) # Clear history from session
    # Optionally: Implement logic here to delete uploaded files associated with this session
    # if they are stored on the persistent disk and linked to the session ID.
    # This would require tracking uploads per session.
    logger.info(f"Chat history cleared for session ID: {session.sid if hasattr(session, 'sid') else 'N/A'}")
    flash("Chat history cleared.", "info")
    return redirect(url_for('index2'))


# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 Not Found error for URL: {request.url}")
    return render_template('404.html'), 404 # Optional: Create a templates/404.html page

@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    logger.warning(f"Request entity too large (limit: {max_size_mb:.0f}MB). URL: {request.url}")
    flash(f"Uploaded file(s) are too large. Maximum allowed total size is {max_size_mb:.0f}MB.", "danger")
    return redirect(url_for('index2'))

@app.errorhandler(500)
def internal_error(error):
    # Log the detailed error. The logger.error in the main try/except blocks
    # should catch most app errors, but this handles unexpected framework errors.
    logger.error(f"Internal Server Error (500): {error}", exc_info=True)
    # Avoid flashing detailed errors to the user in production
    flash("An internal server error occurred. Please try again later or contact support.", "danger")
    return redirect(url_for('index2')) # Redirect to avoid showing error page directly


# --- Main Execution Guard ---
# This block is primarily for LOCAL development.
# Render uses the 'gunicorn app2:app' command specified in the dashboard.
if __name__ == "__main__":
    # Check if essential local dev config is missing
    if not app.config.get('SECRET_KEY') or app.config.get('SECRET_KEY') == 'a_very_secret_key_for_local_dev_only':
        logger.critical("WARNING: Running locally with insecure default SECRET_KEY. Set FLASK_SECRET_KEY environment variable.")
    if not app.ai_model:
         logger.critical("WARNING: Running locally but AI Model failed to initialize. Check GEMINI_API_KEY environment variable and logs.")

    # Use host='0.0.0.0' to be accessible on your local network
    # Debug mode is controlled by FLASK_DEBUG env var via Configuration class
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=app.config['DEBUG'])
