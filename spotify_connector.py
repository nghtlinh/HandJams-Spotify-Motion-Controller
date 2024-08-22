import requests
from spotipy.oauth2 import SpotifyOAuth

from constants import SCOPE

class spotify_api:
    """Class represents connection with Spotify API
    """
    def __init__(self, client_id: str = "", 
                client_secret: str = "", 
                token: str = "", 
                redirect_uri: str = "http://localhost:8888/spotify-api/callback/") -> None:
        """
        Initializes the Spotify connector.

        Args:
            client_id (str, optional): The unique identifier for your Spotify application.
            client_secret (str, optional): The secret key used to authorize your application with the Spotify Web API.
            token (str, optional): The authorization token used to access Spotify
            redirect_uri (str, optional): The URI to which users are redirected after they approve the authorization request.
        """
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        
    def get_token(self) -> str:
        """
        Retrieves and sets the authorization token.

        Returns:
            str: The access token obtained from Spotify
        """
        try:
            oauth = SpotifyOAuth(client_id = self.client_id, 
                                 client_secret = self.client_secret,
                                 redirect_uri = self.redirect_uri,
                                 scope = SCOPE)
            self.token = oauth.get_access_token()['access_token']
            return self.token
        except Exception as e:
            print(f"Exception: {e}")
            
    def get_playback_state(self) -> dict:
        """
        Retrieves the current playback state of user's Spotify account.

        Returns:
            dict: _description_
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/player"
            response = requests.get(endpoint, headers = self.headers)
            info = response.json()
            return info
        except Exception as e:
            print(f"Error retrieving playback state: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def skip_to_previous(self) -> None:
        """
        Skips to the previous track in the user's Spotify playback
        """
        try: 
            endpoint = "https://api.spotify.com/v1/me/player/previous"
            requests.post(endpoint, headers = self.headers)
        except Exception as e:
            print(f"Error skipping to previous track: {e}")
            print("Please ensure Spotify is open and music is playing, or restart the app.")
            
    def skip_to_next(self) -> None:
        """
        Skips to the next track in the user's Spotify playback.
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/player/next"
            requests.post(endpoint, headers = self.headers)
        except Exception as e:
            print(f"Error skipping to next track: {e}")
            print("Please ensure Spotify is open and music is playing.")

    def change_playing_status(self) -> None:
        """
        Toggles the playback status on Spotify. 
        If music is playing, it pauses the playback; if music is paused, it resumes playback.
        """
        try:
            is_playing = self.get_playback_state()['is_playing']
            
            if is_playing:
                self.pause_playback()
            else:
                self.resume_playback()
        except Exception as e:
            print(f"Error changing playback status: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def pause_playback(self) -> None:
        """
        Pauses the currently playing track on Spotify.
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/player/pause"
            requests.put(endpoint, headers = self.headers)
        except Exception as e:
            print(f"Error pausing playback: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def resume_playback(self) -> None:
        """
        Resumes playback of the currently paused track on Spotify
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/player/play"
            requests.put(endpoint, headers = self.headers)
        except Exception as e:
            print(f"Error pausing playback: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def volume_down(self) -> None:
        """
        Decreases the current volume by 5%
        """
        percent = 5
        try:
            current_volume = self.get_playback_state()['device']['volume_precent']
            volume = max(current_volume - percent, 0)
            endpoint = "https://api.spotify.com/v1/me/player/volume?volume_percent="
            requests.put(f"{endpoint}{volume}", headers=self.headers)
        except Exception as e:
            print(f"Error lowering volume: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def volume_up(self) -> None:
        """
        Increases the current volume by 5%
        """
        percent = 5
        try:
            current_volume = self.get_playback_state()['device']['volume_precent']
            volume = max(current_volume + percent, 100)
            endpoint = "https://api.spotify.com/v1/me/player/volume?volume_percent="
            requests.put(f"{endpoint}{volume}", headers=self.headers)
        except Exception as e:
            print(f"Error raising volume: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def fav_track(self) -> None:
        """
        Toggles the currently playing track's saved status. If the track is already in the user's
        saved tracks, it is removed. If not, add the track.
        """
        try:
            saved_tracks_id_list = self.get_saved_tracks_id()
            current_track_id = self.get_playback_state()['item']['id']
            
            # Save or remove the track from the saved songs list
            if current_track_id in saved_tracks_id_list:
                self.removed_track(current_track_id)
            else:
                self.save_track(current_track_id)
        except Exception as e:
            print(f"Error toggling favorite track status: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def get_saved_tracks_id(self) -> list[str]:
        """
        Retrieves a list of track IDs from the user's saved tracks.

        Returns:
            list[str]: _description_
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/tracks"
            response = requests.get(endpoint, headers = self.headers)
            items - response.json()['items']
            
            # Create a list of IDs from the saved tracks
            items_id = [e['track']['id'] for e in items]
            return items_id
        except Exception as e:
            print(f"Error retrieving saved tracks: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def save_track(self, current_track_id: str) -> None:
        """
        Saves the currently playing track to the user's saved tracks list.

        Args:
            current_track_id (str): The ID of the currently playing track.
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/tracks?ids="
            requests.put(f"{endpoint}{current_track_id}", headers = self.headers)
        except Exception as e:
            print(f"Error saving track: {e}")
            print("Please ensure Spotify is open and music is playing.")
            
    def remove_track(self, current_track_id: str) -> None:
        """
        Removes the currently playing track to the user's saved tracks list.

        Args:
            current_track_id (str): The ID of the currently playing track.
        """
        try:
            endpoint = "https://api.spotify.com/v1/me/tracks?ids="
            requests.deletet(f"{endpoint}{current_track_id}", headers = self.headers)
        except Exception as e:
            print(f"Error removing track: {e}")
            print("Please ensure Spotify is open and music is playing.")