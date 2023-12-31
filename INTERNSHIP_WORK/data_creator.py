#
# Copyright 2014 Google Inc. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

"""Performs requests to the Google Maps Distance Matrix API."""

import googlemaps
from googlemaps import convert
import json
from json import *

import pandas as pd
import datetime as DT

def distance_matrix(client, origins, destinations,
                    mode=None, language=None, avoid=None, units=None,
                    departure_time=None, arrival_time=None, transit_mode=None,
                    transit_routing_preference=None, traffic_model=None, region=None):
    """ Gets travel distance and time for a matrix of origins and destinations.

    :param origins: One or more addresses, Place IDs, and/or latitude/longitude
        values, from which to calculate distance and time. Each Place ID string
        must be prepended with 'place_id:'. If you pass an address as a string,
        the service will geocode the string and convert it to a
        latitude/longitude coordinate to calculate directions.
    :type origins: a single location, or a list of locations, where a
        location is a string, dict, list, or tuple

    :param destinations: One or more addresses, Place IDs, and/or lat/lng values
        , to which to calculate distance and time. Each Place ID string must be
        prepended with 'place_id:'. If you pass an address as a string, the
        service will geocode the string and convert it to a latitude/longitude
        coordinate to calculate directions.
    :type destinations: a single location, or a list of locations, where a
        location is a string, dict, list, or tuple

    :param mode: Specifies the mode of transport to use when calculating
        directions. Valid values are "driving", "walking", "transit" or
        "bicycling".
    :type mode: string

    :param language: The language in which to return results.
    :type language: string

    :param avoid: Indicates that the calculated route(s) should avoid the
        indicated features. Valid values are "tolls", "highways" or "ferries".
    :type avoid: string

    :param units: Specifies the unit system to use when displaying results.
        Valid values are "metric" or "imperial".
    :type units: string

    :param departure_time: Specifies the desired time of departure.
    :type departure_time: int or datetime.datetime

    :param arrival_time: Specifies the desired time of arrival for transit
        directions. Note: you can't specify both departure_time and
        arrival_time.
    :type arrival_time: int or datetime.datetime

    :param transit_mode: Specifies one or more preferred modes of transit.
        This parameter may only be specified for requests where the mode is
        transit. Valid values are "bus", "subway", "train", "tram", "rail".
        "rail" is equivalent to ["train", "tram", "subway"].
    :type transit_mode: string or list of strings

    :param transit_routing_preference: Specifies preferences for transit
        requests. Valid values are "less_walking" or "fewer_transfers".
    :type transit_routing_preference: string

    :param traffic_model: Specifies the predictive travel time model to use.
        Valid values are "best_guess" or "optimistic" or "pessimistic".
        The traffic_model parameter may only be specified for requests where
        the travel mode is driving, and where the request includes a
        departure_time.

    :param region: Specifies the prefered region the geocoder should search
        first, but it will not restrict the results to only this region. Valid
        values are a ccTLD code.
    :type region: string

    :rtype: matrix of distances. Results are returned in rows, each row
        containing one origin paired with each destination.
    """

    params = {
        "origins": convert.location_list(origins),
        "destinations": convert.location_list(destinations)
    }

    if mode:
        # NOTE(broady): the mode parameter is not validated by the Maps API
        # server. Check here to prevent silent failures.
        if mode not in ["driving", "walking", "bicycling", "transit"]:
            raise ValueError("Invalid travel mode.")
        params["mode"] = mode

    if language:
        params["language"] = language

    if avoid:
        if avoid not in ["tolls", "highways", "ferries"]:
            raise ValueError("Invalid route restriction.")
        params["avoid"] = avoid

    if units:
        params["units"] = units

    if departure_time:
        params["departure_time"] = convert.time(departure_time)

    if arrival_time:
        params["arrival_time"] = convert.time(arrival_time)

    if departure_time and arrival_time:
        raise ValueError("Should not specify both departure_time and"
                         "arrival_time.")

    if transit_mode:
        params["transit_mode"] = convert.join_list("|", transit_mode)

    if transit_routing_preference:
        params["transit_routing_preference"] = transit_routing_preference

    if traffic_model:
        params["traffic_model"] = traffic_model

    if region:
        params["region"] = region

    return client._request("/maps/api/distancematrix/json", params)



key="AIzaSyApCDXCL7ELIu4HsLET8KFdKTUNrt2n8yc"
client=googlemaps.Client(key)

origins=["Cary,NC,USA", "Durham,NC,USA"]
destinations=["Chapel Hill,NC,USA", "Fayettevile,NC,USA"]
mode=["driving", "walking", "transit","bicycling"]


#dataTable = {"Origin":[],"Destination":[],"Method":[],"Length":[], "Duration":[]}
df = pd.DataFrame()
df2 = pd.DataFrame()

for o in origins:
    for d in destinations:
        for m in mode:
            temp= distance_matrix(client,origins=o,destinations=d, mode=m)

            level0 = ["distance"]
            level1 = ["value"]
            level2 = ["duration"]

            for l in level0:
                for item in level1:
                    for item2 in level2:

                        #print("Going from",o,"to",d,"by",m,"\n Characteristics: \t Length:",temp["rows"][0]["elements"][0][l][item],"Duration:",temp["rows"][0]["elements"][0][item2][item], "\n \n \n")
                        length_of_trip = temp["rows"][0]["elements"][0][l][item]
                        duration_of_trip = temp["rows"][0]["elements"][0][item2][item]

                        duration_in_minutes = round(temp["rows"][0]["elements"][0][item2][item] / 60 , 3)
                        length_in_miles = round(3.28084 * temp["rows"][0]["elements"][0][l][item] / 5280, 3)

                        env_cost=0
                        if m == "walking" or m == "bicycling":
                            env_cost = 0
                        elif m == "driving":
                            env_cost = round(length_in_miles * 0.77,3)
                        else:
                            env_cost = round(0.59 * length_in_miles, 3)


                        leg = o + " to " + d


                        cost=0
                        if m == "walking" or m == "bicycling":
                            cost = 0
                        elif m == "driving":
                            cost = round(length_in_miles * 0.66,3)
                        else:
                            cost = round(0.0625 * length_in_miles, 3)


                        new_row_list = { "start" : o, "end" : d, "modes" : m, "length" : length_in_miles, "time" : duration_in_minutes, "cost":cost, "env_cost" : env_cost , "leg": leg}
                        df = pd.concat([df, pd.DataFrame([new_row_list])], ignore_index=True)

            #find ways to print values of temp we're interested in
            #read values from json with a built in function
            #potentially wjson or rjson

print(df)

df.to_csv("/workspace/PQB/INTERNSHIP_WORK/data/maps_data.csv")