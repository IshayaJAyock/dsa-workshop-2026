"""Travel tools for notebook 3 — weather, distance, exchange rates, and places."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any

import requests

from src.utils import get_env

CURATED_PLACES: dict[str, list[dict[str, str]]] = {
    "kigali,rwanda": [
        {
            "name": "Kigali Genocide Memorial",
            "short_note": "A respectful centre for remembrance and learning.",
            "why_visit": "Understand Rwanda's recent history and reconciliation efforts.",
            "data_scientist_angle": "Study how memorial institutions preserve data, testimony, and public memory.",
        },
        {
            "name": "Kigali Innovation City area",
            "short_note": "Hub for startups, universities, and tech programmes.",
            "why_visit": "See how Rwanda is investing in digital skills and innovation.",
            "data_scientist_angle": "Explore local innovation ecosystems and digital transformation projects.",
        },
        {
            "name": "Mount Kigali viewpoint",
            "short_note": "Panoramic views over the city.",
            "why_visit": "Get orientation and a sense of urban layout.",
            "data_scientist_angle": "Useful for thinking about urban mobility and city analytics.",
        },
    ],
    "accra,ghana": [
        {
            "name": "Kwame Nkrumah Memorial Park",
            "short_note": "Landmark dedicated to Ghana's first president.",
            "why_visit": "Learn about Ghana's independence history.",
            "data_scientist_angle": "Connect civic history with public data and national development narratives.",
        },
        {
            "name": "Labadi Beach area",
            "short_note": "Popular coastal area near the city centre.",
            "why_visit": "Experience local culture and leisure spaces.",
            "data_scientist_angle": "Tourism analytics and coastal urban planning use cases.",
        },
        {
            "name": "Osu / Oxford Street area",
            "short_note": "Busy commercial and social district.",
            "why_visit": "Good for food, shopping, and people-watching.",
            "data_scientist_angle": "Retail activity and urban mobility patterns in a dense district.",
        },
    ],
    "nairobi,kenya": [
        {
            "name": "Nairobi National Museum",
            "short_note": "Museum covering culture, history, and natural heritage.",
            "why_visit": "Compact introduction to Kenyan history and biodiversity.",
            "data_scientist_angle": "Environmental and biodiversity data storytelling.",
        },
        {
            "name": "Kazuri Beads workshop area",
            "short_note": "Social enterprise known for handmade ceramics.",
            "why_visit": "See local craft and community enterprise.",
            "data_scientist_angle": "Social enterprise metrics and fair-workplace case studies.",
        },
        {
            "name": "KICC viewpoint",
            "short_note": "Iconic tower with city views.",
            "why_visit": "Orientation and skyline perspective.",
            "data_scientist_angle": "Urban growth, transport, and infrastructure analytics.",
        },
    ],
    "lagos,nigeria": [
        {
            "name": "National Museum Lagos",
            "short_note": "Museum with Nigerian art and cultural artefacts.",
            "why_visit": "Cultural context for Nigeria's diverse heritage.",
            "data_scientist_angle": "Cultural data preservation and digitisation themes.",
        },
        {
            "name": "Lekki Conservation Centre",
            "short_note": "Nature reserve with canopy walkway.",
            "why_visit": "Green space within a major metropolis.",
            "data_scientist_angle": "Climate, environment, and urban ecology data angles.",
        },
        {
            "name": "Victoria Island business district",
            "short_note": "Commercial hub with offices and restaurants.",
            "why_visit": "See one of Lagos's major economic centres.",
            "data_scientist_angle": "Fintech, digital business, and urban economic activity.",
        },
    ],
    "cape town,south africa": [
        {
            "name": "Table Mountain cableway area",
            "short_note": "Famous viewpoint over the city and coastline.",
            "why_visit": "Iconic Cape Town experience.",
            "data_scientist_angle": "Tourism flow and environmental monitoring use cases.",
        },
        {
            "name": "V&A Waterfront",
            "short_note": "Harbour-side area with shops and museums.",
            "why_visit": "Accessible introduction to the city waterfront.",
            "data_scientist_angle": "Tourism analytics and mixed-use urban development.",
        },
        {
            "name": "District Six Museum",
            "short_note": "Museum about forced removals and community memory.",
            "why_visit": "Important social history in a compact visit.",
            "data_scientist_angle": "Social data, displacement, and policy impact analysis.",
        },
    ],
    "cairo,egypt": [
        {
            "name": "Egyptian Museum area",
            "short_note": "Major museum district for ancient Egyptian history.",
            "why_visit": "Essential cultural introduction to Cairo.",
            "data_scientist_angle": "Archaeological cataloguing and heritage digitisation.",
        },
        {
            "name": "Khan el-Khalili bazaar",
            "short_note": "Historic market area.",
            "why_visit": "Experience old Cairo commerce and craft traditions.",
            "data_scientist_angle": "Informal economy and market activity patterns.",
        },
        {
            "name": "Nile Corniche",
            "short_note": "Riverfront promenade through the city.",
            "why_visit": "Easy walk with views of the Nile.",
            "data_scientist_angle": "Urban mobility and river-city infrastructure.",
        },
    ],
    "marrakech,morocco": [
        {
            "name": "Jemaa el-Fnaa",
            "short_note": "Famous central square and gathering place.",
            "why_visit": "Vibrant cultural heart of the medina.",
            "data_scientist_angle": "Tourism analytics and informal street economy.",
        },
        {
            "name": "Majorelle Garden area",
            "short_note": "Well-known garden and design landmark.",
            "why_visit": "Calmer cultural stop within the city.",
            "data_scientist_angle": "Cultural tourism and visitor flow management.",
        },
        {
            "name": "Medina souks",
            "short_note": "Traditional market lanes.",
            "why_visit": "Crafts, food, and local commerce.",
            "data_scientist_angle": "Small-business activity and local supply chains.",
        },
    ],
    "dakar,senegal": [
        {
            "name": "African Renaissance Monument area",
            "short_note": "Large landmark overlooking the city.",
            "why_visit": "Orientation and modern Dakar symbolism.",
            "data_scientist_angle": "Urban landmarks and public infrastructure planning.",
        },
        {
            "name": "Île de Gorée day-trip reference",
            "short_note": "Historic island often visited from Dakar.",
            "why_visit": "Important site for Atlantic history and remembrance.",
            "data_scientist_angle": "Historical records, migration, and memorial data contexts.",
        },
        {
            "name": "Plateau district",
            "short_note": "Administrative and business centre.",
            "why_visit": "Useful base for navigating the city.",
            "data_scientist_angle": "Public administration and digital government themes.",
        },
    ],
    "addis ababa,ethiopia": [
        {
            "name": "National Museum of Ethiopia",
            "short_note": "Home to important archaeological exhibits.",
            "why_visit": "Key cultural stop in the capital.",
            "data_scientist_angle": "Archaeology, heritage science, and cataloguing systems.",
        },
        {
            "name": "Meskel Square area",
            "short_note": "Major public square and urban landmark.",
            "why_visit": "Central reference point in the city.",
            "data_scientist_angle": "Urban public space and transport planning.",
        },
        {
            "name": "Entoto Park area",
            "short_note": "Hilltop park with views over Addis Ababa.",
            "why_visit": "Green space and city panorama.",
            "data_scientist_angle": "Urban green infrastructure and environmental monitoring.",
        },
    ],
    "kampala,uganda": [
        {
            "name": "Uganda Museum",
            "short_note": "National museum on culture and history.",
            "why_visit": "Compact introduction to Ugandan heritage.",
            "data_scientist_angle": "Cultural preservation and public education data.",
        },
        {
            "name": "Kasubi Tombs area",
            "short_note": "Important cultural site linked to Buganda kingdom history.",
            "why_visit": "Significant heritage landmark.",
            "data_scientist_angle": "Heritage conservation and community record-keeping.",
        },
        {
            "name": "Ndere Cultural Centre",
            "short_note": "Centre for music, dance, and performance.",
            "why_visit": "Accessible cultural evening option.",
            "data_scientist_angle": "Creative industries and cultural event analytics.",
        },
    ],
}


def _city_key(city: str, country: str) -> str:
    return f"{city.strip().lower()},{country.strip().lower()}"


def geocode_city(city: str, country: str | None = None) -> dict[str, Any]:
    """Use Open-Meteo geocoding API to get coordinates and timezone."""
    query = f"{city},{country}" if country else city
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ValueError(f"Geocoding request failed for {query}: {exc}") from exc
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not geocode city: {query}")
    top = results[0]
    return {
        "city": top.get("name", city),
        "country": top.get("country", country or ""),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone", "UTC"),
    }


def get_weather(
    latitude: float,
    longitude: float,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Use Open-Meteo forecast API for a simple weather summary."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
        "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min",
        "timezone": timezone or "auto",
        "forecast_days": 1,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    current = data.get("current", {})
    daily = data.get("daily", {})
    code = current.get("weather_code", 0)
    return {
        "temperature": current.get("temperature_2m"),
        "weather_summary": _weather_code_to_text(code),
        "precipitation_probability": (
            daily.get("precipitation_probability_max", [None])[0]
        ),
        "wind_speed": current.get("wind_speed_10m"),
        "date": daily.get("time", [""])[0],
        "units": {
            "temperature": data.get("current_units", {}).get("temperature_2m", "°C"),
            "wind_speed": data.get("current_units", {}).get("wind_speed_10m", "km/h"),
        },
    }


def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        80: "Rain showers",
        95: "Thunderstorm",
    }
    return mapping.get(code, "Variable conditions")


def calculate_distance_km(
    base_latitude: float,
    base_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> dict[str, Any]:
    """Return approximate straight-line distance in kilometres."""
    try:
        from geopy.distance import geodesic

        distance_km = geodesic(
            (base_latitude, base_longitude),
            (destination_latitude, destination_longitude),
        ).kilometers
    except Exception:
        distance_km = _haversine_km(
            base_latitude,
            base_longitude,
            destination_latitude,
            destination_longitude,
        )
    return {
        "distance_km": round(distance_km, 1),
        "note": "Approximate straight-line distance, not flight route distance.",
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def get_exchange_rate(
    base_currency: str,
    destination_currency: str,
    amount: float = 1.0,
) -> dict[str, Any]:
    """Use Frankfurter API for exchange-rate lookup."""
    base = base_currency.strip().upper()
    dest = destination_currency.strip().upper()
    if len(base) != 3 or len(dest) != 3 or not base.isalpha() or not dest.isalpha():
        return {
            "error": (
                f"Invalid currency code(s): '{base_currency}' / '{destination_currency}'. "
                "Use three-letter codes such as RWF and GHS."
            )
        }

    url = f"https://api.frankfurter.dev/v2/rate/{base}/{dest}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {
            "error": (
                f"Could not fetch {base}/{dest} exchange rate. "
                f"Check the codes and try again. ({exc})"
            )
        }

    rate = data.get("rate")
    converted = round(amount * rate, 4) if rate else None
    return {
        "base_currency": base,
        "destination_currency": dest,
        "rate": rate,
        "amount": amount,
        "converted_amount": converted,
        "date": data.get("date"),
    }


def _search_places_online(city: str, country: str) -> list[dict[str, str]] | None:
    query = f"top places to visit in {city} {country}"
    tavily_key = get_env("TAVILY_API_KEY")
    serp_key = get_env("SERPAPI_API_KEY")

    if tavily_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 3},
                timeout=20,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            places = []
            for item in results[:3]:
                places.append(
                    {
                        "name": item.get("title", "Unknown place")[:80],
                        "short_note": item.get("content", "")[:160],
                        "why_visit": "Suggested by web search result.",
                        "data_scientist_angle": (
                            "Check whether tourism, mobility, or public data themes appear in local projects."
                        ),
                    }
                )
            if places:
                return places
        except requests.RequestException:
            pass

    if serp_key:
        try:
            response = requests.get(
                "https://serpapi.com/search",
                params={"engine": "google", "q": query, "api_key": serp_key},
                timeout=20,
            )
            response.raise_for_status()
            organic = response.json().get("organic_results", [])
            places = []
            for item in organic[:3]:
                places.append(
                    {
                        "name": item.get("title", "Unknown place")[:80],
                        "short_note": item.get("snippet", "")[:160],
                        "why_visit": "Suggested by web search result.",
                        "data_scientist_angle": (
                            "Look for local innovation, analytics, or research-linked attractions."
                        ),
                    }
                )
            if places:
                return places
        except requests.RequestException:
            pass

    return None


def get_top_places_to_visit(city: str, country: str) -> list[dict[str, str]]:
    """Return top 3 places using search APIs or curated fallback."""
    online = _search_places_online(city, country)
    if online:
        return online[:3]

    curated = CURATED_PLACES.get(_city_key(city, country))
    if curated:
        return curated

    return [
        {
            "name": f"{city} city centre",
            "short_note": "Start with the central district for orientation.",
            "why_visit": "Useful first stop when curated data is unavailable.",
            "data_scientist_angle": "Urban mobility and local services mapping.",
        },
        {
            "name": f"Local museum or cultural centre in {city}",
            "short_note": "Check official listings for opening hours.",
            "why_visit": "Cultural context for visitors.",
            "data_scientist_angle": "Public education and cultural data themes.",
        },
        {
            "name": f"Public viewpoint or park in {city}",
            "short_note": "A calm place to get oriented.",
            "why_visit": "Low-risk first outing while travelling.",
            "data_scientist_angle": "Green infrastructure and environmental monitoring.",
        },
    ]


def gather_travel_tool_results(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    base_currency: str,
    destination_currency: str,
    amount: float = 100.0,
) -> dict[str, Any]:
    """
    Run all travel tools and return one evidence dictionary.

    This is the heart of 'manual tool calling':
    the app gathers facts first, then asks the LLM to explain them.
    """
    base_geo = geocode_city(base_city, base_country)
    destination_geo = geocode_city(destination_city, destination_country)
    weather = get_weather(
        destination_geo["latitude"],
        destination_geo["longitude"],
        destination_geo["timezone"],
    )
    distance = calculate_distance_km(
        base_geo["latitude"],
        base_geo["longitude"],
        destination_geo["latitude"],
        destination_geo["longitude"],
    )
    exchange = get_exchange_rate(base_currency, destination_currency, amount=amount)
    places = get_top_places_to_visit(destination_city, destination_country)
    return {
        "base_geo": base_geo,
        "destination_geo": destination_geo,
        "weather": weather,
        "distance": distance,
        "exchange": exchange,
        "places": places,
    }
