import requests
import json


def google_programmable_search(query, api_key, search_engine_id):
    """Performs a search using the Google Programmable Search Engine API.

    Args:
        query: The search query string.
        api_key: Your Google API key.
        search_engine_id: The ID of your Programmable Search Engine.

    Returns:
        A list of search results (dictionaries), or None if there was an error.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 5,  # Get 5 results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "items" in data:
            return data["items"]
        else:
            print("No results found.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")  # More descriptive error
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")  # More descriptive error
        return None


# Replace with your actual API key and search engine ID
MY_API_KEY = "AIzaSyDSFs3nmrUwzpBHPMAOdlEyBEzutPu60nA"
MY_SEARCH_ENGINE_ID = "640638fe26c87482f"

search_term = "emla contraindicaciones"
results = google_programmable_search(search_term, MY_API_KEY, MY_SEARCH_ENGINE_ID)

if results:
    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"  Title: {result['title']}")
        print(f"  Link: {result['link']}")
        print(f"  Snippet: {result['snippet']}")
        print("-" * 20)
