from requests import Request
import requests

# This will be used to authenticate with the FTX host via API-Key
def authenticate(TYPE, SUBURL):
    # Type: 'GET' or 'POST' (string)
    # SubUrl: The api data that the user wants to call (string, e. g. /markets)

    request = Request(TYPE, SUBURL)
    prepared = request.prepare()
    prepared.headers['Content-Type'] = 'application/json'

    s = requests.Session()

    return s.send(prepared).json()
