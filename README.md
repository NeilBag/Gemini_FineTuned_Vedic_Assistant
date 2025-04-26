# Gemini Fine-Tuned Vedic Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Choose your license -->

A Flask-based web application providing an interface to interact with a Google Gemini 1.5 Pro model that has been **fine-tuned specifically on data related to the Hinduism Vedas**. This application allows users to ask questions and receive contextually relevant answers based on the model's specialized training.

The application maintains chat history for conversational context, supports optional file uploads (text, documents) to provide additional context for prompts, and allows exporting the AI's last response. It is designed with deployment on platforms like Render or Replit in mind.

## Features

*   **Web-Based Chat Interface:** Clean and modern UI built with Flask and Bootstrap 5.
*   **Fine-Tuned Gemini Model:** Utilizes a Gemini 1.5 Pro model (`tunedModels/hinduism-veda-expert-v1`) fine-tuned on Vedic scriptures data for specialized knowledge.
*   **Conversational Context:** Maintains chat history per user session to allow for follow-up questions.
*   **File Uploads:** Users can optionally upload files (`.txt`, `.docx`, `.pdf`, images) to provide context alongside their text prompts. (Note: PDF content extraction is currently basic/not fully implemented).
*   **Response Export:** Export the last AI-generated response to `.docx` or `.pdf` format.
*   **Clear History:** Option to clear the current chat session history.
*   **Persistent Storage:** Uses file-based storage for chat history (and uploads), designed to work with persistent disks on platforms like Render or the standard filesystem on Replit.
*   **Deployment Ready:** Configured for deployment using Gunicorn on PaaS platforms (Render, Replit examples provided).

## Technology Stack

*   **Programming Language:** Python 3.11+
*   **Web Framework:** Flask
*   **AI Model:** Google Gemini 1.5 Pro (Fine-tuned)
*   **AI Library:** `google-generativeai`
*   **Frontend:** HTML, CSS, JavaScript, Bootstrap 5
*   **WSGI Server:** Gunicorn (for deployment)
*   **File Handling:** `python-docx` (for .docx reading), `Pillow` (for image info), `xhtml2pdf` (for PDF export)
*   **Environment Management:** `python-dotenv` (for local development)
*   **Deployment Platforms:** Render, Replit (or similar Python PaaS)

## Project Structure

                                                 
*(Note: `.env` file for local secrets is intentionally not listed as it should be in `.gitignore`)*

## Setup and Installation (Local Development)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeilBag/Gemini_FineTuned_Vedic_Assistant.git
    cd Gemini_FineTuned_Vedic_Assistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Add the following lines, replacing the placeholder values:
        ```dotenv
        # .env file (This file should NOT be committed to Git)

        # Mandatory: Your Google AI API Key
        GEMINI_API_KEY=YOUR_ACTUAL_GOOGLE_AI_API_KEY

        # Mandatory: Generate a strong secret key for Flask session signing
        # Use: python -c 'import secrets; print(secrets.token_hex(24))' to generate one
        FLASK_SECRET_KEY=YOUR_GENERATED_STRONG_SECRET_KEY

        # Optional: Set log level (DEBUG, INFO, WARNING, ERROR)
        LOG_LEVEL=INFO

        # Optional: Set Flask environment (development or production)
        # FLASK_ENV=development

        # Optional: Override the default fine-tuned model name
        # GEMINI_MODEL_NAME=tunedModels/hinduism-veda-expert-v1
        ```
    *   **Important:** Obtain your `GEMINI_API_KEY` from Google AI Studio or Google Cloud Console. Keep it secret!
    *   **Important:** Generate a unique and strong `FLASK_SECRET_KEY`. Do not use the default fallback from the code.

5.  **Create Local Directories (if they don't exist):**
    *   The application attempts to create these, but you can ensure they exist:
    ```bash
    mkdir uploads
    mkdir chat_histories
    ```

## Running Locally

Once the setup is complete, run the Flask development server: The application will be accessible at http://127.0.0.1:5000 (or the port specified if PORT env var is set).

```bash
python app2.py

Fine-Tuning Context
The core AI model used, identified as tunedModels/hinduism-veda-expert-v1 (or as overridden by GEMINI_MODEL_NAME), is a Google Gemini 1.5 Pro model that has undergone fine-tuning.
Training Data: The fine-tuning process utilized question/answer pairs derived from Hinduism Vedas, stored in Hinduism_Vedas_qa_generated2.json.
Training Script: The script FineTune_Gemini_Hindu_Vedas.py contains the logic used for performing the fine-tuning via the Google AI API (this script is not run as part of the web application itself).
This specialization allows the model to provide more accurate and contextually appropriate answers related to Vedic knowledge compared to the base Gemini model.
Usage
Access the application via its URL (local or deployed).
Enter your question or prompt related to Hinduism Vedas in the text area.
(Optional) Click "Choose Files" to upload relevant documents (.txt, .docx, .pdf, images) that might provide extra context for your prompt. The content of text/docx files will be pre-pended to your prompt.
Click "Send" (or press Enter in the text area).
The AI's response will appear in the chat history.
Ask follow-up questions. The application maintains context within your session.
To export the last AI response, select "DOCX" or "PDF" from the "Export" dropdown and click "Go".
To start a fresh conversation, click "Clear Chat History".
