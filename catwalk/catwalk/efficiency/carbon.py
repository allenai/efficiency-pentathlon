# -*- coding: utf-8 -*-
"""
Only tested in Python 3.
You may need to install the "requests" Python3 module.

Be sure to fill in your username, password, org name and email before running
"""

from os import path
import requests
from requests.auth import HTTPBasicAuth

# account details
USERNAME = "haop"
PASSWORD = "8x.qYrY,)2rK"
EMAIL = "haop@allenai.org"
ORG = "AI2"

# request details
BA = "CAISO_NORTH"  # Seattile City Light: SCL # identify grid region

# starttime and endtime are optional, if ommited will return the latest value
START = "2020-03-01T00:00:00-0000"  # UTC offset of 0 (PDT is -7, PST -8)
END = "2020-03-01T00:45:00-0000"


def register(username, password, email, org):
    url = "https://api2.watttime.org/register"
    params = {"username": username,
              "password": password,
              "email": email,
              "org": org}
    rsp = requests.post(url, json=params)
    print(rsp.text)


def login(username, password):
    url = "https://api2.watttime.org/login"
    try:
        rsp = requests.get(url, auth=HTTPBasicAuth(username, password))
    except BaseException as e:
        print("There was an error making your login request: {}".format(e))
        return None

    try:
        token = rsp.json()["token"]
    except BaseException:
        print("There was an error logging in. The message returned from the "
              "api is {}".format(rsp.text))
        return None

    return token


def data(token, ba, starttime, endtime):
    url = "https://api2.watttime.org/data"
    headers = {"Authorization": "Bearer {}".format(token)}
    params = {"ba": ba, "starttime": starttime, "endtime": endtime}

    rsp = requests.get(url, headers=headers, params=params)
    # print(rsp.text)  # uncomment to see raw response
    return rsp.json()


def index(token, ba):
    url = "https://api2.watttime.org/index"
    headers = {"Authorization": "Bearer {}".format(token)}
    params = {"ba": ba}

    rsp = requests.get(url, headers=headers, params=params)
    # print(rsp.text)  # uncomment to see raw response
    return rsp.json()


def forecast(token, ba, starttime=None, endtime=None):
    url = "https://api2.watttime.org/forecast"
    headers = {"Authorization": "Bearer {}".format(token)}
    params = {"ba": ba}
    if starttime:
        params.update({"starttime": starttime, "endtime": endtime})

    rsp = requests.get(url, headers=headers, params=params)
    # print(rsp.text)  # uncomment to see raw response
    return rsp.json()


def historical(token, ba):
    url = "https://api2.watttime.org/historical"
    headers = {"Authorization": "Bearer {}".format(token)}
    params = {"ba": ba}
    rsp = requests.get(url, headers=headers, params=params)
    cur_dir = path.dirname(path.realpath(__file__))
    file_path = path.join(cur_dir, "{}_historical.zip".format(ba))
    with open(file_path, "wb") as fp:
        fp.write(rsp.content)

    print("Wrote historical data for {} to {}".format(ba, file_path))


def get_regions(token):
    list_url = "https://api2.watttime.org/v2/ba-access"
    headers = {"Authorization": "Bearer {}".format(token)}
    params = {"all": "true"}
    rsp=requests.get(list_url, headers=headers, params=params)
    return rsp.text


def get_realtime_moer(region = BA):
    token = login(USERNAME, PASSWORD)
    if not token:
        print("You will need to fix your login credentials (username and password "
            "at the start of this file) before you can query other endpoints. "
            "Make sure that you have registered at least once by uncommenting "
            "the register(username, password, email, org) line near the bottom "
            "of this file.")
        exit()
    realtime_index = index(token, region)
    return realtime_index


def get_realtime_carbon(energy, region = BA):
    """
        Energy: in Wh
        region: only north CA is supported without subscription.
                Emerald landing's region code should br SCL
    """
    moer = float(get_realtime_moer(BA)["moer"])  # in lbs/MWh
    def lbs2kg(lbs):
        return lbs / 2.205
    carbon = lbs2kg(energy * 1e-3 * moer)  # in g
    return carbon


if __name__ == "__main__":
    # Only register once!!
    # register(USERNAME, PASSWORD, EMAIL, ORG)

    token = login(USERNAME, PASSWORD)
    if not token:
        print("You will need to fix your login credentials (username and password "
            "at the start of this file) before you can query other endpoints. "
            "Make sure that you have registered at least once by uncommenting "
            "the register(username, password, email, org) line near the bottom "
            "of this file.")
        exit()
    realtime_index = index(token, BA)
    print(realtime_index["ba"])
    print(realtime_index["moer"])  # CO2: lbs/MWh
