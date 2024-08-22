from configparser import ConfigParser
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog
from core.spotify_api import spotify_api
from pyui.auth_window import Ui_AuthorizationWindow

class auth_window(QDialog, Ui_AuthorizationWindow):
    """
    Authorization window for Spotify login.
    """

    def __init__(self) -> None:
        """
        Initialize the UI and connect buttons.
        """
        super().__init__()
        self.setupUi(self)
        self._connect_buttons()
        
    def _connect_buttons(self) -> None:
        """
        Connect the submit button to its handler.
        """
        self.submit_btn.clicked.connect(self._on_login_button_clicked)
    
    def create_submit_message(self, text: str = "", color: str = "1db954;") -> None:
        """
        Displays a message on the UI and sets its color.

        Args:
            text (str): The message text to display.
            color (str): The color of the message text (default is green).
        """
        self.submit_msg.setText(text)
        self.submit_msg.setStyleSheet(f"color: {color}")
        
    def check_connection(self):
        """
        Authenticate with Spotify and handle the result.
        """
        self.create_submit_message("Please wait...", "#bbb")
        client_id = self.entered_client_id.text()
        client_secret = self.entered_client_secret.text()

        # Initialize spotify_account with client_id and client_secret
        spotify_account = spotify_api(client_id=client_id, client_secret=client_secret)
        
        # Now retrieve the token
        token = spotify_account.get_token()
        if token:
            self.create_submit_message("Login successful!", "#1db954")
            self.save_config(client_id, client_secret, token)
        else:
            self.create_submit_message("Login failed. Please try again.", "red")
            self.entered_client_id.setText("")
            self.entered_client_secret.setText("")
            
    
    def save_config(self, client_id: str, client_secret: str, token: str) -> None:
        """
        Save client credentials and token to config file.
        """
        config = ConfigParser()
        config.read('config.ini')
        
        if not config.has_section('CLIENT'):
            config.add_section('CLIENT')
        
        config.set('CLIENT', 'client_id', client_id)
        config.set('CLIENT', 'client_secret', client_secret)
        config.set('CLIENT', 'token', token)
        
        with open('config.ini', 'w') as file:
            config.write(file)
            
    
    def _on_login_button_clicked(self) -> None:
        """
        Handle the submit button click.
        """
        self.check_connection()