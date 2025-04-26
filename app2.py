import os
import logging
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from docx import Document
# Consider potential issues with xhtml2pdf dependencies, ensure installed correctly
try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None # Handle missing dependency gracefully if needed
import google.generativeai as genai
from dotenv import load_dotenv # For loading API key from .env file
import tempfile # For unique temporary files
import traceback # For detailed error logging

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Configuration:
    def __init__(self):
        self.DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't') # Read debug from env
        self.SECRET_KEY = os.getenv('FLASK_SECRET_KEY', os.urandom(24)) # Read secret key from env or generate one
        self.UPLOAD_FOLDER = 'uploads'
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx', 'mp4', 'avi', 'mov'} # Consider if all these are processable
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
        self.TEMP_FOLDER = 'temp' # Will use tempfile module instead for exports

class GenerativeAISettings:
    def __init__(self):
        # Consider making these configurable via environment variables too
        self.MODEL_NAME = "tunedModels/hinduism-veda-expert-v1" # Updated model name, check latest available
        self.GENERATION_CONFIG = {
            "temperature": 0.7, # Adjusted for potentially more creative output
            "top_p": 0.10,
            "top_k": 64,
            "max_output_tokens": 128000, # More realistic token limit for Flash models
            "response_mime_type": "text/plain",
        }

def create_app(config=Configuration()):
    app = Flask(__name__)
    app.config.from_object(config)

    # Initialize Generative AI
    api_key = os.getenv("GEMINI_API_KEY") # <<< FIXED: Load from environment
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        # Depending on deployment, you might raise ValueError or handle differently
        # For now, let it proceed but log the error. The AI call will fail later.
        # raise ValueError("GEMINI_API_KEY environment variable not set")
    else:
        try:
            genai.configure(api_key=api_key)
            # Store model in app context? For now, create per request or keep global?
            # Let's keep it accessible via app context or a dedicated holder if needed
            # For simplicity now, we'll initialize it globally below create_app
        except Exception as e:
            logger.exception(f"Failed to configure Generative AI: {e}")
            # Handle configuration error appropriately

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # TEMP_FOLDER defined in config isn't used directly for exports anymore

    return app

app = create_app()

# Initialize the AI model (consider lazy loading or placing within app context if preferred)
model = None
if os.getenv("GEMINI_API_KEY"):
    try:
        model = genai.GenerativeModel(
            model_name=GenerativeAISettings().MODEL_NAME,
            generation_config=GenerativeAISettings().GENERATION_CONFIG
        )
    except Exception as e:
        logger.exception(f"Failed to initialize GenerativeModel: {e}")
else:
    logger.warning("AI Model not initialized because GEMINI_API_KEY is not set.")


# --- File Handling ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def secure_save_file(file):
    """Saves the file securely and returns its path and original filename."""
    if not file or not file.filename:
        return None, None
    filename = secure_filename(file.filename)
    if not filename: # Handle cases where secure_filename returns empty string
        filename = "unnamed_file" # Or generate a unique name
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        return file_path, file.filename # Return original filename too if needed
    except Exception as e:
        logger.exception(f"Failed to save file {filename}: {e}")
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
        # Add more handlers here (e.g., PDF using PyMuPDF or PyPDF2)
        # elif extension == '.pdf':
        #     # Requires additional libraries like PyMuPDF (fitz)
        #     import fitz # Example using PyMuPDF
        #     doc = fitz.open(file_path)
        #     content = "".join(page.get_text() for page in doc)
        #     doc.close()
        else:
            logger.warning(f"Content reading not supported for extension: {extension}")
            return f"(File type {extension} not processed for content)"

        # Limit content size to avoid overly large prompts
        max_chars = 10000 # Example limit, adjust as needed
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
        # Ensure entries have the expected structure
        if 'user' in entry and entry['user'] is not None:
            history.append({"role": "user", "parts": [{"text": str(entry['user'])}]})
        if 'model' in entry and entry['model'] is not None:
            history.append({"role": "model", "parts": [{"text": str(entry['model'])}]})
    return history

def build_history_for_template(session_history):
    """Builds history suitable for rendering in the template."""
    # Currently same as session_history, but could be adapted if needed
    # Ensure entries have 'user' and 'model' keys for the template
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
        # Use tempfile for unique, auto-cleaned file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            docx_path = temp_file.name
            document.save(docx_path)
        logger.info(f"Generated DOCX: {docx_path}")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            pdf_path = temp_file.name
            # Simple HTML wrapping, <pre> preserves whitespace and line breaks
            # Consider more robust HTML conversion if needed
            html_content = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body><pre>{text}</pre></body></html>"
            # Use BytesIO buffer for pisa
            from io import BytesIO
            pdf_buffer = BytesIO()
            pisa_status = pisa.CreatePDF(BytesIO(html_content.encode('UTF-8')), dest=pdf_buffer)

            if pisa_status.err:
                 logger.error(f"Error creating PDF using pisa: {pisa_status.err}")
                 return None

            # Write buffer content to the temp file
            with open(pdf_path, "wb") as f:
                f.write(pdf_buffer.getvalue())

        logger.info(f"Generated PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.exception(f"Error creating PDF file: {e}")
        return None


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index2():
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Ensure model is initialized
    if model is None:
        flash("AI Model is not configured. Please set the GEMINI_API_KEY environment variable.", "error")
        # Render template without attempting AI interaction
        template_history = build_history_for_template(session['chat_history'])
        return render_template("index2.html", history=template_history)


    if request.method == "POST":
        try:
            prompt = request.form.get("prompt", "").strip()
            export_format = request.form.get("export_format", "") # Check if export button was clicked

            # --- File Upload Processing ---
            uploaded_file_contents = []
            uploaded_filenames = []
            files = request.files.getlist("files")
            if files:
                for file in files:
                    if file and file.filename and allowed_file(file.filename):
                        file_path, original_filename = secure_save_file(file)
                        if file_path:
                            logger.info(f"Processing uploaded file: {original_filename} (saved as {os.path.basename(file_path)})")
                            content = read_file_content(file_path)
                            uploaded_file_contents.append(f"--- Content from {original_filename} ---\n{content}\n--- End of {original_filename} ---")
                            uploaded_filenames.append(original_filename)
                        else:
                            flash(f"Could not save file: {file.filename}", "warning")
                    elif file and file.filename:
                        flash(f"File type not allowed: {file.filename}", "warning")

            # --- Combine Prompt and File Content ---
            full_prompt = prompt
            if uploaded_file_contents:
                # Prepend file contents to the user's prompt
                full_prompt = "\n\n".join(uploaded_file_contents) + "\n\n" + prompt
                logger.info(f"Combined prompt includes content from files: {', '.join(uploaded_filenames)}")

            if not prompt and not uploaded_file_contents: # Require either text prompt or uploaded file
                flash("Please enter a prompt or upload a file.", "warning")
                # Re-render the page without calling AI
                template_history = build_history_for_template(session['chat_history'])
                return render_template("index2.html", history=template_history)

            # --- Call Generative AI ---
            logger.info("Sending prompt to Generative AI...")
            api_history = build_history_for_api(session['chat_history'])
            chat_session = model.start_chat(history=api_history)
            response = chat_session.send_message(full_prompt) # Send combined prompt
            response_text = response.text
            logger.info("Received response from Generative AI.")

            # --- Update Session History ---
            # Store the original user prompt, not the combined one, for cleaner history display?
            # Or store the combined prompt? Let's store the original prompt + note about files.
            history_user_entry = prompt
            if uploaded_filenames:
                 history_user_entry += f"\n\n(Used uploaded files: {', '.join(uploaded_filenames)})"

            session['chat_history'].append({'user': history_user_entry, 'model': response_text})
            session.modified = True # Important to mark session as modified

            # --- Handle Export ---
            if export_format:
                export_path = None
                mimetype = None
                download_name = None

                if export_format == "docx":
                    export_path = export_docx(response_text)
                    mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    download_name="response.docx"
                elif export_format == "pdf":
                    if pisa: # Only attempt if pisa is available
                        export_path = export_pdf(response_text)
                        mimetype='application/pdf'
                        download_name="output.pdf"
                    else:
                        flash("PDF export is currently unavailable.", "error")
                        # Fall through to re-render the page

                if export_path and mimetype and download_name:
                    try:
                        return send_file(export_path, mimetype=mimetype, as_attachment=True, download_name=download_name)
                    finally:
                        # Clean up the temporary file after sending
                        try:
                            os.remove(export_path)
                            logger.info(f"Cleaned up temporary file: {export_path}")
                        except OSError as e:
                            logger.error(f"Error removing temporary file {export_path}: {e}")
                elif not export_path and export_format == "pdf" and not pisa:
                    pass # Already flashed message
                else:
                    flash(f"Failed to export as {export_format}.", "error")
                 # Fall through to re-render the page if export failed

            # --- Display Response: Re-render template --- <<< FIXED
            template_history = build_history_for_template(session['chat_history'])
            # No redirect here, render the template directly to show the new response
            return render_template("index2.html", history=template_history)

        except Exception as e:
            logger.error(f"Error processing POST request: {str(e)}")
            logger.error(traceback.format_exc()) # Log full traceback for debugging
            flash(f"An error occurred while processing your request: {str(e)}", "error")
            # Redirect to clear the POST state and show the error
            return redirect(url_for('index2'))

    # --- Handle GET Request ---
    # Build history for rendering the template
    template_history = build_history_for_template(session['chat_history'])
    return render_template("index2.html", history=template_history)


@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop('chat_history', None)
    flash("Chat history cleared.", "info")
    return redirect(url_for('index2'))


@app.errorhandler(413)
def request_entity_too_large(e):
    flash("The uploaded file(s) are too large. Maximum allowed total size is 16MB.", "error")
    return redirect(url_for('index2')) # <<< FIXED Typo


if __name__ == "__main__":
    # Set host='0.0.0.0' to be accessible externally, default is '127.0.0.1'
    # Debug mode should ideally be False in production (read from config)
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])