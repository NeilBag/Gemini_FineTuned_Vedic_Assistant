import unittest
from unittest.mock import patch, MagicMock
import os

# Add the directory containing app2.py to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Assuming test_app2.py is in the same directory as app2.py

import app2
from app2 import app # Flask app instance

class TestApp2SendMessage(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False # Disable CSRF for testing forms
        app.config['SECRET_KEY'] = 'test_secret_key' # Set a secret key for session
        self.client = app.test_client()
        # Set a dummy API key for testing initialization
        os.environ['GEMINI_API_KEY'] = 'test_api_key'
        # Re-initialize model in app2 with the test key (if app2 logic allows for re-init or uses env var directly)
        # This might require adjusting app2.py to re-initialize the model if API key changes,
        # or ensuring the model is initialized after this env var is set.
        # For now, assume app2.model will pick this up or use a patch.

    @patch('app2.genai.GenerativeModel')
    def test_send_message_structure(self, MockGenerativeModel):
        # Mock the model instance and its methods
        mock_model_instance = MockGenerativeModel.return_value
        mock_chat_session = MagicMock()
        mock_model_instance.start_chat.return_value = mock_chat_session
        mock_chat_session.send_message.return_value = MagicMock(text="Test response")

        # Ensure app2.model is this mock_model_instance
        # This is important if app2.model is initialized at import time.
        with patch('app2.model', mock_model_instance):
            test_prompt = "Hello, world!"
            response = self.client.post('/', data={'prompt': test_prompt})

            # Assert that send_message was called
            mock_chat_session.send_message.assert_called_once()

            # Assert the structure of the content passed to send_message
            args, kwargs = mock_chat_session.send_message.call_args
            sent_content = kwargs.get('content') # As per our change: content={"parts": ...}
            
            self.assertIsNotNone(sent_content, "Content was not passed to send_message")
            self.assertIn('parts', sent_content, "Content should have 'parts' key")
            self.assertIsInstance(sent_content['parts'], list, "'parts' should be a list")
            self.assertTrue(len(sent_content['parts']) > 0, "'parts' list should not be empty")
            self.assertIn('text', sent_content['parts'][0], "First part should have 'text' key")
            self.assertEqual(sent_content['parts'][0]['text'], test_prompt, "Text in the first part is not the same as the prompt")

if __name__ == '__main__':
    unittest.main()
