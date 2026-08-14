import math
import requests

REF_LAT = 30.1645096
REF_LNG = 74.1767266

URL = "http://65.2.122.207:3000/get-detections"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def fetch_json(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def find_closest_object(data):
    closest = None
    min_distance = float("inf")

    radar_list = data.get("data", {}).get("radar", [])
    rf_list = data.get("data", {}).get("rf", [])

    for obj in radar_list:
        latlng = obj.get("rdr_track_latlng")
        if latlng:
            try:
                lat, lng = map(float, latlng.split(","))
                dist = haversine(REF_LAT, REF_LNG, lat, lng) * 1000

                if dist < min_distance:
                    min_distance = dist
                    closest = {
                        "source": "RADAR",
                        "object": obj,
                        "drone_distance": dist,
                        "azimuth": obj.get("rdr_track_azimuth"),
                        "latitude": lat,
                        "longitude": lng
                    }
            except:
                pass

    for obj in rf_list:
        latlng = obj.get("rfdf_dect_drone_location")
        if latlng and latlng != "0,0":
            try:
                lat, lng = map(float, latlng.split(","))
                dist = haversine(REF_LAT, REF_LNG, lat, lng) * 1000

                if dist < min_distance:
                    min_distance = dist
                    closest = {
                        "source": "RF",
                        "object": obj,
                        "drone_distance": dist,
                        "azimuth": obj.get("rfdf_dect_Azimuth"),
                        "latitude": lat,
                        "longitude": lng
                    }
            except:
                pass

    return closest


def get_IDMS_tgt_data(Lat = 30.1645096, Long = 74.1767266):
    global REF_LAT, REF_LNG
    REF_LAT= Lat
    REF_LNG = Long

    data = fetch_json(URL)

    if not data:
        return None

    result = find_closest_object(data)

    if not result:
        return None

    return result

def main():
    result = get_IDMS_tgt_data()
    print(result)

'''
if __name__ == "__main__":
    main()
'''