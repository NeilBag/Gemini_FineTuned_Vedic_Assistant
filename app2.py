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

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Configuration:
    def __init__(self):
        # General Flask settings
        self.DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
        # IMPORTANT: Set FLASK_SECRET_KEY in Render Environment Variables
        self.SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_key_for_local_dev_only') # Fallback for local dev
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

        # --- Paths for Render Persistent Disks (with local fallbacks) ---
        # Assumes disks named 'flask-sessions' and 'uploads' are mounted at
        # /mnt/data/flask_session and /mnt/data/uploads respectively in Render.
        # Render sets env vars like RENDER_DISK_PATH_<disk_name> automatically.
        # Use these paths if running on Render, otherwise use local relative paths.

        # Path for Flask-Session files
        self.SESSION_FILE_DIR = os.getenv('RENDER_DISK_PATH_FLASK_SESSION', './.flask_session')

        # Path for user file uploads
        self.UPLOAD_FOLDER = os.getenv('RENDER_DISK_PATH_UPLOADS', 'uploads')
        # --- End Persistent Disk Paths ---

        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx'} # Removed video for simplicity now

        # Flask-Session specific config
        self.SESSION_TYPE = 'filesystem' # Use filesystem for sessions
        self.SESSION_PERMANENT = False # Session expires when browser closes (or configure lifetime)
        self.SESSION_USE_SIGNER = True # Sign the session cookie identifier (needs SECRET_KEY)


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
        logger.info(f"Session directory set to: {app.config['SESSION_FILE_DIR']}")
    except OSError as e:
        logger.error(f"Could not create session directory {app.config['SESSION_FILE_DIR']}: {e}")
        # Handle error appropriately - perhaps raise an exception or exit if critical

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info(f"Upload directory set to: {app.config['UPLOAD_FOLDER']}")
    except OSError as e:
        logger.error(f"Could not create upload directory {app.config['UPLOAD_FOLDER']}: {e}")
        # Handle error appropriately

    # --- Initialize Flask-Session ---
    # Initialize AFTER config is loaded and session directory exists
    Session(app)

    # --- Initialize Generative AI ---
    if not ai_settings.API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set.")
        # Store None or a placeholder? App will likely fail later if None.
        app.ai_model = None
    else:
        try:
            genai.configure(api_key=ai_settings.API_KEY)
            app.ai_model = genai.GenerativeModel(
                model_name=ai_settings.MODEL_NAME,
                generation_config=ai_settings.GENERATION_CONFIG
            )
            logger.info(f"Generative AI model '{ai_settings.MODEL_NAME}' initialized.")
        except Exception as e:
            logger.exception(f"Failed to configure or initialize Generative AI: {e}")
            app.ai_model = None # Ensure it's None if init fails

    return app

app = create_app()
model = app.ai_model # Convenience reference to the model stored in app context

# --- File Handling ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def secure_save_file(file):
    """Saves the file securely and returns its path and original filename."""
    if not file or not file.filename:
        return None, None
    filename = secure_filename(file.filename)
    if not filename:
        filename = "unnamed_upload" # Or generate a unique name
    # Use the configured upload folder path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        return file_path, file.filename
    except Exception as e:
        logger.exception(f"Failed to save file {filename} to {file_path}: {e}")
        return None, None

def read_file_content(file_path):
    """Reads content from supported text-based files."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    content = ""
    try:
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif extension == '.docx':
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
        # Basic image handling example (just confirms it's an image)
        elif extension in ['.png', '.jpg', '.jpeg'] and Image:
             try:
                 img = Image.open(file_path)
                 content = f"(Image file detected: {os.path.basename(file_path)}, format: {img.format}, size: {img.size})"
                 img.close()
             except Exception as img_e:
                 logger.warning(f"Could not process image file {file_path}: {img_e}")
                 content = f"(Could not fully process image file: {os.path.basename(file_path)})"
        elif extension == '.pdf':
             # Placeholder: Add PDF text extraction here if needed (e.g., using PyMuPDF)
             content = f"(PDF file detected: {os.path.basename(file_path)} - content extraction not implemented)"
        else:
            logger.warning(f"Content reading not implemented for extension: {extension}")
            return f"(File type {extension} not processed for content)"

        # Limit content size to avoid overly large prompts
        max_chars = 10000 # Adjust as needed
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"
        return content

    except Exception as e:
        logger.exception(f"Error reading file content from {file_path}: {e}")
        return f"(Error reading file {os.path.basename(file_path)})"


# --- History Handling ---
def build_history_for_api(session_history):
    """Builds history in the format required by the Google Generative AI API."""
    history = []
    for entry in session_history:
        if 'user' in entry and entry['user'] is not None:
            # Ensure parts contains a list, even if just one text part
            history.append({"role": "user", "parts": [{"text": str(entry['user'])}]})
        if 'model' in entry and entry['model'] is not None:
            # Ensure parts contains a list
            history.append({"role": "model", "parts": [{"text": str(entry['model'])}]})
    return history

def build_history_for_template(session_history):
    """Builds history suitable for rendering in the template."""
    template_history = []
    for entry in session_history:
        template_history.append({
            'user': entry.get('user'),
            'model': entry.get('model')
        })
    return template_history


# --- Export Functions ---
def export_docx(text):
    """Exports text to a DOCX file using a temporary file."""
    try:
        document = Document()
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
    try:
        # Use tempfile for unique, auto-cleaned file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as temp_file:
            pdf_path = temp_file.name
            # Simple HTML wrapping, <pre> preserves whitespace and line breaks
            html_content = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Export</title></head><body><pre>{text}</pre></body></html>"

            # Use BytesIO buffer for pisa
            from io import BytesIO
            pdf_buffer = BytesIO()
            pisa_status = pisa.CreatePDF(BytesIO(html_content.encode('UTF-8')), dest=pdf_buffer)

            if pisa_status.err:
                logger.error(f"Error creating PDF using pisa: {pisa_status.err}")
                return None

            # Write buffer content to the temp file
            temp_file.write(pdf_buffer.getvalue()) # Write directly to the NamedTemporaryFile object

        logger.info(f"Generated PDF for export: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.exception(f"Error creating PDF file: {e}")
        return None


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index2():
    # Initialize chat history in session if not present
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Check if AI model is available
    current_model = app.ai_model # Get model from app context
    if current_model is None:
        flash("AI Model is not configured or failed to initialize. Please check server logs and API key.", "danger")
        template_history = build_history_for_template(session['chat_history'])
        return render_template("index2.html", history=template_history)

    if request.method == "POST":
        try:
            action = request.form.get("action", "submit") # Distinguish submit vs export clicks
            prompt = request.form.get("prompt", "").strip()
            export_format = request.form.get("export_format", "")

            # --- File Upload Processing ---
            uploaded_file_contents = []
            uploaded_filenames = []
            files = request.files.getlist("files")
            if files:
                for file in files:
                    if file and file.filename and allowed_file(file.filename):
                        file_path, original_filename = secure_save_file(file)
                        if file_path:
                            logger.info(f"Processing uploaded file: {original_filename}")
                            content = read_file_content(file_path)
                            uploaded_file_contents.append(f"--- Content from {original_filename} ---\n{content}\n--- End of {original_filename} ---")
                            uploaded_filenames.append(original_filename)
                            # Note: We are not deleting uploaded files automatically here.
                            # Consider adding cleanup logic if needed, maybe after session ends or periodically.
                        else:
                            flash(f"Could not save file: {secure_filename(file.filename)}", "warning")
                    elif file and file.filename:
                        flash(f"File type not allowed or file error: {secure_filename(file.filename)}", "warning")

            # --- Handle Prompt Submission ---
            if action == "submit":
                # --- Combine Prompt and File Content ---
                full_prompt = prompt
                if uploaded_file_contents:
                    full_prompt = "\n\n".join(uploaded_file_contents) + "\n\nUser Prompt:\n" + prompt
                    logger.info(f"Combined prompt includes content from files: {', '.join(uploaded_filenames)}")

                if not prompt and not uploaded_file_contents:
                    flash("Please enter a prompt or upload a file.", "warning")
                    return redirect(url_for('index2')) # Redirect GET after POST pattern

                # --- Call Generative AI ---
                logger.info("Sending prompt to Generative AI...")
                api_history = build_history_for_api(session['chat_history'])
                try:
                    chat_session = current_model.start_chat(history=api_history)
                    response = chat_session.send_message(full_prompt)
                    response_text = response.text
                    logger.info("Received response from Generative AI.")
                except Exception as ai_error:
                     logger.exception(f"Error during Generative AI call: {ai_error}")
                     flash(f"Error communicating with AI: {ai_error}", "danger")
                     return redirect(url_for('index2'))


                # --- Update Session History ---
                history_user_entry = prompt
                if uploaded_filenames:
                    history_user_entry += f"\n\n(Used uploaded files: {', '.join(uploaded_filenames)})"

                session['chat_history'].append({'user': history_user_entry, 'model': response_text})
                session.modified = True # Mark session as modified

                return redirect(url_for('index2')) # Redirect after POST success

            # --- Handle Export Request ---
            elif action == "export":
                if not export_format:
                    flash("Please select an export format (DOCX or PDF).", "warning")
                    return redirect(url_for('index2'))

                # Get the text of the *last* model response from history
                last_model_response = None
                if session.get('chat_history'):
                    # Iterate backwards to find the last entry with a 'model' key
                    for entry in reversed(session['chat_history']):
                        if 'model' in entry and entry['model'] is not None:
                            last_model_response = entry['model']
                            break

                if not last_model_response:
                    flash("No AI response found in history to export.", "warning")
                    return redirect(url_for('index2'))

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
                        # Use send_file and ensure cleanup happens after request
                        return send_file(export_path, mimetype=mimetype, as_attachment=True, download_name=download_name)
                    finally:
                        # Schedule cleanup (safer than immediate remove)
                        # This might require a background task runner in a real prod app,
                        # but for simple cases, immediate remove after send *might* work.
                        # For robustness, tempfile usually handles cleanup on its own eventually,
                        # especially if the server process exits cleanly.
                        try:
                            os.remove(export_path)
                            logger.info(f"Cleaned up temporary export file: {export_path}")
                        except OSError as e:
                            logger.error(f"Error removing temporary export file {export_path}: {e}")
                else:
                    flash(f"Failed to generate export file for format {export_format}.", "danger")
                    return redirect(url_for('index2'))

            else:
                 # Unknown action
                 flash("Invalid form action.", "warning")
                 return redirect(url_for('index2'))


        except Exception as e:
            logger.error(f"Error processing POST request: {str(e)}")
            logger.error(traceback.format_exc()) # Log full traceback
            flash(f"An unexpected error occurred: {str(e)}", "danger")
            return redirect(url_for('index2')) # Redirect on error

    # --- Handle GET Request ---
    # Build history for rendering the template
    template_history = build_history_for_template(session.get('chat_history', []))
    return render_template("index2.html", history=template_history)


@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop('chat_history', None)
    flash("Chat history cleared.", "info")
    # Optionally: Clean up uploaded files associated with this session if tracked
    return redirect(url_for('index2'))


@app.errorhandler(413)
def request_entity_too_large(e):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    flash(f"Uploaded file(s) are too large. Maximum allowed total size is {max_size_mb:.0f}MB.", "danger")
    return redirect(url_for('index2'))

# --- Main Execution ---
# This block is primarily for LOCAL development.
# Render uses the 'gunicorn app2:app' command specified in the dashboard.
if __name__ == "__main__":
    # Use host='0.0.0.0' to be accessible on your local network
    # Debug should be False if FLASK_DEBUG env var is not 'true'
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
