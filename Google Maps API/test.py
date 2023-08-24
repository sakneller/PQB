import unittest
import codecs

from urllib.parse import urlparse, parse_qsl


class TestCase(unittest.TestCase):
    def assertURLEqual(self, first, second, msg=None):
        """Check that two arguments are equivalent URLs. Ignores the order of
        query arguments.
        """
        first_parsed = urlparse(first)
        second_parsed = urlparse(second)
        self.assertEqual(first_parsed[:3], second_parsed[:3], msg)

        first_qsl = sorted(parse_qsl(first_parsed.query))
        second_qsl = sorted(parse_qsl(second_parsed.query))
        self.assertEqual(first_qsl, second_qsl, msg)

    def u(self, string):
        """Create a unicode string, compatible across all versions of Python."""
        # NOTE(cbro): Python 3-3.2 does not have the u'' syntax.
        return codecs.unicode_escape_decode(string)[0]

from datetime import datetime
import time

import responses

import googlemaps
from . import TestCase


class DistanceMatrixTest(TestCase):
    def setUp(self):
        self.key = "AIzaasdf"
        self.client = googlemaps.Client(self.key)

    @responses.activate
    def test_basic_params(self):
        responses.add(
            responses.GET,
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            body='{"status":"OK","rows":[]}',
            status=200,
            content_type="application/json",
        )

        origins = [
            "Perth, Australia",
            "Sydney, Australia",
            "Melbourne, Australia",
            "Adelaide, Australia",
            "Brisbane, Australia",
            "Darwin, Australia",
            "Hobart, Australia",
            "Canberra, Australia",
        ]
        destinations = [
            "Uluru, Australia",
            "Kakadu, Australia",
            "Blue Mountains, Australia",
            "Bungle Bungles, Australia",
            "The Pinnacles, Australia",
        ]

        matrix = self.client.distance_matrix(origins, destinations)

        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual(
            "https://maps.googleapis.com/maps/api/distancematrix/json?"
            "key=%s&origins=Perth%%2C+Australia%%7CSydney%%2C+"
            "Australia%%7CMelbourne%%2C+Australia%%7CAdelaide%%2C+"
            "Australia%%7CBrisbane%%2C+Australia%%7CDarwin%%2C+"
            "Australia%%7CHobart%%2C+Australia%%7CCanberra%%2C+Australia&"
            "destinations=Uluru%%2C+Australia%%7CKakadu%%2C+Australia%%7C"
            "Blue+Mountains%%2C+Australia%%7CBungle+Bungles%%2C+Australia"
            "%%7CThe+Pinnacles%%2C+Australia" % self.key,
            responses.calls[0].request.url,
        )

    @responses.activate
    def test_mixed_params(self):
        responses.add(
            responses.GET,
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            body='{"status":"OK","rows":[]}',
            status=200,
            content_type="application/json",
        )

        origins = [
            "Bobcaygeon ON", [41.43206, -81.38992],
            "place_id:ChIJ7cv00DwsDogRAMDACa2m4K8"
        ]
        destinations = [
            (43.012486, -83.6964149),
            {"lat": 42.8863855, "lng": -78.8781627},
        ]

        matrix = self.client.distance_matrix(origins, destinations)

        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual(
            "https://maps.googleapis.com/maps/api/distancematrix/json?"
            "key=%s&origins=Bobcaygeon+ON%%7C41.43206%%2C-81.38992%%7C"
            "place_id%%3AChIJ7cv00DwsDogRAMDACa2m4K8&"
            "destinations=43.012486%%2C-83.6964149%%7C42.8863855%%2C"
            "-78.8781627" % self.key,
            responses.calls[0].request.url,
        )

    @responses.activate
    def test_all_params(self):
        responses.add(
            responses.GET,
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            body='{"status":"OK","rows":[]}',
            status=200,
            content_type="application/json",
        )

        origins = [
            "Perth, Australia",
            "Sydney, Australia",
            "Melbourne, Australia",
            "Adelaide, Australia",
            "Brisbane, Australia",
            "Darwin, Australia",
            "Hobart, Australia",
            "Canberra, Australia",
        ]
        destinations = [
            "Uluru, Australia",
            "Kakadu, Australia",
            "Blue Mountains, Australia",
            "Bungle Bungles, Australia",
            "The Pinnacles, Australia",
        ]

        now = datetime.now()
        matrix = self.client.distance_matrix(
            origins,
            destinations,
            mode="driving",
            language="en-AU",
            avoid="tolls",
            units="imperial",
            departure_time=now,
            traffic_model="optimistic",
        )

        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual(
            "https://maps.googleapis.com/maps/api/distancematrix/json?"
            "origins=Perth%%2C+Australia%%7CSydney%%2C+Australia%%7C"
            "Melbourne%%2C+Australia%%7CAdelaide%%2C+Australia%%7C"
            "Brisbane%%2C+Australia%%7CDarwin%%2C+Australia%%7CHobart%%2C+"
            "Australia%%7CCanberra%%2C+Australia&language=en-AU&"
            "avoid=tolls&mode=driving&key=%s&units=imperial&"
            "destinations=Uluru%%2C+Australia%%7CKakadu%%2C+Australia%%7C"
            "Blue+Mountains%%2C+Australia%%7CBungle+Bungles%%2C+Australia"
            "%%7CThe+Pinnacles%%2C+Australia&departure_time=%d"
            "&traffic_model=optimistic" % (self.key, time.mktime(now.timetuple())),
            responses.calls[0].request.url,
        )

    @responses.activate
    def test_lang_param(self):
        responses.add(
            responses.GET,
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            body='{"status":"OK","rows":[]}',
            status=200,
            content_type="application/json",
        )

        origins = ["Vancouver BC", "Seattle"]
        destinations = ["San Francisco", "Victoria BC"]

        matrix = self.client.distance_matrix(
            origins, destinations, language="fr-FR", mode="bicycling"
        )

        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual(
            "https://maps.googleapis.com/maps/api/distancematrix/json?"
            "key=%s&language=fr-FR&mode=bicycling&"
            "origins=Vancouver+BC%%7CSeattle&"
            "destinations=San+Francisco%%7CVictoria+BC" % self.key,
            responses.calls[0].request.url,
        )
    @responses.activate
    def test_place_id_param(self):
        responses.add(
            responses.GET,
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            body='{"status":"OK","rows":[]}',
            status=200,
            content_type="application/json",
        )

        origins = [
            'place_id:ChIJ7cv00DwsDogRAMDACa2m4K8',
            'place_id:ChIJzxcfI6qAa4cR1jaKJ_j0jhE',
        ]
        destinations = [
            'place_id:ChIJPZDrEzLsZIgRoNrpodC5P30',
            'place_id:ChIJjQmTaV0E9YgRC2MLmS_e_mY',
        ]

        matrix = self.client.distance_matrix(origins, destinations)

        self.assertEqual(1, len(responses.calls))
        self.assertURLEqual(
            "https://maps.googleapis.com/maps/api/distancematrix/json?"
            "key=%s&"
            "origins=place_id%%3AChIJ7cv00DwsDogRAMDACa2m4K8%%7C"
            "place_id%%3AChIJzxcfI6qAa4cR1jaKJ_j0jhE&"
            "destinations=place_id%%3AChIJPZDrEzLsZIgRoNrpodC5P30%%7C"
            "place_id%%3AChIJjQmTaV0E9YgRC2MLmS_e_mY" % self.key,
            responses.calls[0].request.url,
        )