import json
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable

geolocator = Nominatim(user_agent="skiresortrecommendations")


def locToUrl(loc):
    return str(loc[1]) + "," + str(loc[0])


def metersToMiles(meters):
    return meters / 1609.344


def getDistance(source, locs):
    # source is a tuple (latitude, longitude)
    # locs is a list of tuples (latitude, longitude, area name)
    # returns a list of tuples (distance from source, area name)
    url = "http://router.project-osrm.org/table/v1/driving/" + \
        locToUrl(source)
    for loc in locs:
        url += ";"
        url += locToUrl(loc)
    url += "?sources=0&annotations=distance"
    r = requests.get(url)
    d = json.loads(r.text)
    distances = d['distances'][0][1:]
    return [(metersToMiles(distances[i]), locs[i][2]) for i in range(len(locs))]


with open("dataset/skiing/locations.json", "r") as f:
    d = json.load(f)
    locations = [(lst[0], lst[1], area_name)
                 for area_name, lst in d['area_name_to_location'].items()]


def getDistanceForAreas(query):
    try:
        loc = geolocator.geocode(query)
    except GeocoderUnavailable:
        return {"error": "The Geopy API is currently unavailable. Please try again."}
    if loc is not None:
        source = (loc.latitude, loc.longitude)
        try:
            dists = getDistance(source, locations)
        except json.JSONDecodeError:
            return {'error': "External API error while calculating distance. Please try again."}
        return {area_name: distance for distance, area_name in dists}
    else:
        return {"error": "Location could not be determined."}


def sortAreasByDistance(area_to_distance):
    # area_to_distance is a dict with keys equal to area names
    #   and values equal to distances
    # returns a sorted list of pairs (area, dist) sorted by increasing distance
    result = [(area, dist) for area, dist in area_to_distance.items()]
    return sorted(result, key=lambda x: x[1])


# print(getDistanceForAreas("neasdfasdf"))
