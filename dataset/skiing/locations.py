import json
import requests
from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="skiresortrecommendations")


def getCoordinates(query):
    location = geolocator.geocode(query)
    return (location.latitude, location.longitude)


def getLocations():
    with open("dataset/skiing/reviews.json", "r") as f:
        with open("app/irsystem/models/locations.json", "w") as locations_file:  # TODO change path
            with open("dataset/skiing/manualqueries.json", "r") as f2:
                corrections = json.load(f2)
                locations = {}
                data = json.load(f)
                processed = []
                for review in data['reviews']:
                    if review['area_name'] not in processed:
                        processed.append(review['area_name'])
                        # area_name = review['area_name'] if review['area_name'] not in corrections else corrections[review['area_name']]
                        if review["area_name"] in corrections:
                            query = corrections[review['area_name']]
                        else:
                            query = review["area_name"].replace(
                                "-", " ").replace("usa", "").replace("area", "").replace(" ride", "").strip() + " " + review["state"].replace("-", " ")
                        try:
                            locations[review['area_name']
                                      ] = getCoordinates(query)
                        except AttributeError:
                            print(query, "**",
                                  review['area_name'], review['state'])

                d = {"area_name_to_location": locations}
                json.dump(d, locations_file)


getLocations()
