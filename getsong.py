import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pandas as pd
import time
import os
import statistics
import webbrowser
from dotenv import load_dotenv
import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

def save_token():
    load_dotenv('../.env')
    CLIENT_ID = os.environ.get('CLIENT_ID')
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
    command = f"""echo "CLIENT_ID='{CLIENT_ID}'\nCLIENT_SECRET='{CLIENT_SECRET}'" > .env"""
    os.system(command)

def spotify_token ():
    
    """This function refreshes a 'queries token' for a given app on Spotify.
    queries token are refreshed every 3600 seconds
    returns: token as a string
    """

    # 1. Defining: credentials fot the app
    
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    print(CLIENT_ID)
    
    #2. Request
    
    body_params = {"grant_type":"client_credentials"}
    url = "https://accounts.spotify.com/api/token"
    response = requests.post(url, data=body_params, auth=(CLIENT_ID,CLIENT_SECRET))
    
    try:
        token = response.json()["access_token"]
        return token

    except:
        print("The request did not go through. Maybe check credentials?")


def getsong(seed_song):
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')

    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Define the seed artist and genre
    seed_tracks = []
    seed_tracks.append(seed_song['id'].iloc[0])
    print(seed_tracks)
    #Pop,Hip-Hop,Rock,Indie,R&B,Electronic,Dance,Latin,Alternative,Country,Reggaeton
    seed_genres = ['rock', 'pop', 'hip-hop','electronic']

    # Define the target energy and valence values
    target_energy = float(seed_song['valence'].iloc[0])
    target_valence = float(seed_song['energy'].iloc[0])

    # Call the recommendation API
    results = sp.recommendations(seed_tracks=seed_tracks, seed_genres=seed_genres, target_energy=target_energy, target_valence=target_valence)

    # Extract the recommended tracks from the response
    recommended_tracks = results['tracks']
    #print(recommended_tracks)
    #print(results['tracks'][0]['album']['artists'][0]['external_urls']['spotify'])

    # Print the recommended tracks
    for track in recommended_tracks:
        if 'album' in track and 'artists' in track['album'] and 'external_urls' in track and 'spotify' in track['external_urls']:
            track_info = {
                'name': track['name'],
                'artist': track['album']['artists'][0]['name'],
                'album': track['album']['name'],
                'track_link': track['external_urls']['spotify'],
                'artists': track['album']['artists'],
                'available_markets': track['available_markets'],
                'disc_number': track['disc_number'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'external_ids': track['external_ids'],
                'href': track['href'],
                'id': track['id'],
                'is_local': track['is_local'],
                'popularity': track['popularity'],
                'preview_url': track['preview_url'],
                'track_number': track['track_number'],
                'type': track['type'],
                'uri': track['uri']
            }
            recommended_tracks.append(track_info)
    recommended_tracks_df = pd.DataFrame(recommended_tracks)
    recommended_tracks_df.dropna(subset=['track_link'],inplace=True)
    recommended_tracks_df.track_link.sample()
    link = recommended_tracks_df.track_link.sample().values[0]
    webbrowser.open_new_tab(link)
    return link