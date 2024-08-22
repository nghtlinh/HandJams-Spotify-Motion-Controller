from configparser import ConfigParser
from typing import Any

from PyQt5.QtWidgets import QMainWindow
from gui.auth_window import auth_window
from gui.detection_window import detection_window
from core.spotify_api import spotify_api
from pyui.start_window import Ui_MainWindow


class main_window(QMainWindow, Ui_MainWindow):
    """Main application window for the GUI."""

    def __init__(self) -> None:
        """Initialize the main window, connect buttons, and set up windows."""
        super().__init__()
        self.setupUi(self)
        self._connect_buttons()
        self.path = "config.ini"
        self.login = None
        self.detection_window = detection_window()
        self.auth_window = auth_window()
        
    def _connect_buttons(self) -> None:
        """Connect buttons to their respective actions."""
        self.start_btn.clicked.connect(self._on_start_button_clicked)
        self.login_btn.clicked.connect(self._on_auth_button_clicked)
        
    def _on_auth_button_clicked(self) -> None:
        """Show the authorization window."""
        self.auth_window.show()
        
    def _on_start_button_clicked(self) -> None:
        """Handle the start button click event. Launch detection if logged in, otherwise prompt for login."""
        if self._log_in():
            self.detection_window.show()
        else:
            self._on_auth_button_clicked()
            
    def _log_in(self) -> Any:
        """Attempt to log in using stored credentials.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        client_id, client_secret = self._read_config()
        
        if len(client_id) > 0 and len(client_secret) > 0:
            spotify_instance = spotify_api(client_id=client_id,
                                      client_secret=client_secret)
            if spotify_instance.get_token():
                self.save_token(spotify_instance.token)
                self.login = spotify_instance.token
                self.detection_window.token = spotify_instance.token
            return True
        return False
    
    def save_token(self, token: str) -> None:
        """Save the Spotify API token to the config file."""
        config = ConfigParser()
        config.read(self.path)
        config.set('CLIENT', 'token', token)
        
        with open('config.ini', 'w') as file:
            config.write(file)
            
    def _read_config(self) -> tuple:
        """Read client ID and secret from the config file.

        Returns:
            Tuple[str, str]: The client ID and client secret.
        """        
        parser = ConfigParser()
        parser.read(self.path)
        client_id = parser.get('CLIENT', 'client_id')
        client_secret = parser.get('CLIENT', 'client_secret')
        return client_id, client_secret