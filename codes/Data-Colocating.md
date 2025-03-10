<!-- Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data -->
## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data


### Step 0: Read in Functions Needed

This code is used to connect GoogleDrive into Colab.
```python
from google.colab import drive
drive.mount('/content/drive')
```

This script is used to extract, process, and analyse Sentinel-2 and Sentinel-3 Earth observation data from the Copernicus data space ecosystem.
```python
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point, shape
import numpy as np
import requests
import pandas as pd
from xml.etree import ElementTree as ET
import os
import json
import folium


def make_api_request(url, method="GET", data=None, headers=None):
    global access_token
    if not headers:
        headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.request(method, url, json=data, headers=headers)
    if response.status_code in [401, 403]:
        global refresh_token
        access_token = refresh_access_token(refresh_token)
        headers["Authorization"] = f"Bearer {access_token}"
        response = requests.request(method, url, json=data, headers=headers)
    return response


def query_sentinel3_olci_arctic_data(start_date, end_date, token):
    """
    Queries Sentinel-3 OLCI data within a specified time range from the Copernicus Data Space,
    targeting data collected over the Arctic region.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.

    Returns:
    DataFrame: Contains details about the Sentinel-3 OLCI images.
    """

    all_data = []
    arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"
    # arctic_polygon = (
    #     "POLYGON ((-81.7 71.7, -81.7 73.8, -75.1 73.8, -75.1 71.7, -81.7 71.7))"
    # )

    filter_string = (
        f"Collection/Name eq 'SENTINEL-3' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'OL_1_EFR___') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(next_url, headers=headers)
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def get_access_and_refresh_token(username, password):
    """Retrieve both access and refresh tokens."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    tokens = response.json()
    return tokens["access_token"], tokens["refresh_token"]


def refresh_access_token(refresh_token):
    """Attempt to refresh the access token using the refresh token."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": "cdse-public",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # This will throw an error for non-2xx responses
        return response.json()["access_token"]
    except requests.exceptions.HTTPError as e:
        print(f"Failed to refresh token: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 400:
            print("Refresh token invalid, attempting re-authentication...")
            # Attempt to re-authenticate
            username = username
            password = password
            # This requires securely managing the credentials, which might not be feasible in all contexts
            access_token, new_refresh_token = get_access_and_refresh_token(
                username, password
            )  # This is a placeholder
            refresh_token = (
                new_refresh_token  # Update the global refresh token with the new one
            )
            return access_token
        else:
            raise

def download_single_product(
    product_id, file_name, access_token, download_dir="downloaded_products"
):
    """
    Download a single product from the Copernicus Data Space.

    :param product_id: The unique identifier for the product.
    :param file_name: The name of the file to be downloaded.
    :param access_token: The access token for authorization.
    :param download_dir: The directory where the product will be saved.
    """
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Construct the download URL
    url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    # Set up the session and headers
    headers = {"Authorization": f"Bearer {access_token}"}
    session = requests.Session()
    session.headers.update(headers)

    # Perform the request
    response = session.get(url, headers=headers, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Define the path for the output file
        output_file_path = os.path.join(download_dir, file_name + ".zip")

        # Stream the content to a file
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {output_file_path}")
    else:
        print(
            f"Failed to download product {product_id}. Status Code: {response.status_code}"
        )

def query_sentinel3_sral_arctic_data(start_date, end_date, token):
    """
    Queries Sentinel-3 SRAL data within a specified time range from the Copernicus Data Space,
    targeting data collected over the Arctic region.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.

    Returns:
    DataFrame: Contains details about the Sentinel-3 SRAL images.
    """

    all_data = []
    # arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"
    arctic_polygon = (
        "POLYGON ((-81.7 71.7, -81.7 73.8, -75.1 73.8, -75.1 71.7, -81.7 71.7))"
    )

    filter_string = (
        f"Collection/Name eq 'SENTINEL-3' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'SR_2_LAN_SI') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(
            next_url, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def query_sentinel2_arctic_data(
    start_date,
    end_date,
    token,
    min_cloud_percentage=10,
    max_cloud_percentage=50,
):
    """
    Queries Sentinel-2 data within a specified time range from the Copernicus Data Space,
    considering a range of cloud coverage by treating greater than and less than conditions as separate attributes.
    Handles pagination to fetch all available data.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    token (str): Access token for authentication.
    min_cloud_percentage (int): Minimum allowed cloud coverage.
    max_cloud_percentage (int): Maximum allowed cloud coverage.

    Returns:
    DataFrame: Contains details about the Sentinel-2 images.
    """

    all_data = []
    arctic_polygon = "POLYGON((-180 60, 180 60, 180 90, -180 90, -180 60))"

    filter_string = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value ge {min_cloud_percentage}) and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value le {max_cloud_percentage}) and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.999Z"
    )

    next_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter={filter_string} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{arctic_polygon}')&"
        f"$top=1000"
    )

    headers = {"Authorization": f"Bearer {token}"}

    while next_url:
        response = make_api_request(
            next_url, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            data = response.json()["value"]
            all_data.extend(data)
            next_url = response.json().get("@odata.nextLink")
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break

    return pd.DataFrame(all_data)


def plot_results(results):
    m = folium.Map(location=[0, 0], zoom_start=2)
    for idx, row in results.iterrows():
        try:
            geojson1 = json.loads(row["Satellite1_Footprint"].replace("'", '"'))
            geojson2 = json.loads(row["Satellite2_Footprint"].replace("'", '"'))

            folium.GeoJson(geojson1, name=row["Satellite1_Name"]).add_to(m)
            folium.GeoJson(geojson2, name=row["Satellite2_Name"]).add_to(m)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    folium.LayerControl().add_to(m)
    return m


def parse_geofootprint(footprint):
    """
    Parses a JSON-like string to extract the GeoJSON and convert to a Shapely geometry.
    """
    try:
        geo_json = json.loads(footprint.replace("'", '"'))
        return shape(geo_json)
    except json.JSONDecodeError:
        return None


def check_collocation(
    df1, df2, start_date, end_date, time_window=pd.to_timedelta("1 day")
):

    collocated = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    for idx1, row1 in df1.iterrows():
        footprint1 = parse_geofootprint(row1["GeoFootprint"])
        if footprint1 is None:
            continue

        s1_start = row1["ContentDate.Start"]
        s1_end = row1["ContentDate.End"]

        if s1_end < start_date or s1_start > end_date:
            continue

        s1_start_adjusted = s1_start - time_window
        s1_end_adjusted = s1_end + time_window

        for idx2, row2 in df2.iterrows():
            footprint2 = parse_geofootprint(row2["GeoFootprint"])
            if footprint2 is None:
                continue

            s2_start = row2["ContentDate.Start"]
            s2_end = row2["ContentDate.End"]

            if s2_end < start_date or s2_start > end_date:
                continue
            if max(s1_start_adjusted, s2_start) <= min(s1_end_adjusted, s2_end):
                if footprint1.intersects(footprint2):
                    collocated.append(
                        {
                            "Satellite1_Name": row1["Name"],
                            "Satellite1_ID": row1["Id"],
                            "Satellite1_Footprint": row1["GeoFootprint"],
                            "Satellite2_Name": row2["Name"],
                            "Satellite2_ID": row2["Id"],
                            "Satellite2_Footprint": row2["GeoFootprint"],
                            "Overlap_Start": max(
                                s1_start_adjusted, s2_start
                            ).isoformat(),
                            "Overlap_End": min(s1_end_adjusted, s2_end).isoformat(),
                        }
                    )

    return pd.DataFrame(collocated)


def make_timezone_naive(dt):
    """Convert a timezone-aware datetime object to timezone-naive in local time."""
    return dt.replace(tzinfo=None)
```

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)

This code used to retrieve and save metadata of Sentinel-2 and Sentinel-3 OLCI satellite data within a specified date range.
```python
username = ""
password = ""
access_token, refresh_token = get_access_and_refresh_token(username, password)
start_date = "2018-06-01"
end_date = "2018-06-02"
path_to_save_data = "/content/drive/MyDrive/GEOL0069/2425/Week 4/" # Here you can edit where you want to save your metadata
s3_olci_metadata = query_sentinel3_olci_arctic_data(
    start_date, end_date, access_token
)

s2_metadata = query_sentinel2_arctic_data(
    start_date,
    end_date,
    access_token,
    min_cloud_percentage=0,
    max_cloud_percentage=10,
)

# You can also save the metadata
s3_olci_metadata.to_csv(
    path_to_save_data+"sentinel3_olci_metadata.csv",
    index=False,
)

s2_metadata.to_csv(
    path_to_save_data+"sentinel2_metadata.csv",
    index=False,
)
```

And this code used to display the Sentinel-3 OLCI satellite metadata as a table.
```python
from IPython.display import display

display(s3_olci_metadata)
```
This code used to display the Sentinel-2 metadata as a table.
```python
from IPython.display import display

display(s2_metadata)
```


#### Co-locate the data

Read the metadata.
```python
s3_olci_metadata = pd.read_csv(
    path_to_save_data + "sentinel3_olci_metadata.csv"
)
s2_metadata = pd.read_csv(
    path_to_save_data + "sentinel2_metadata.csv"
)
```

Process the metadata.
```python
s3_olci_metadata["ContentDate.Start"] = pd.to_datetime(
    s3_olci_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s3_olci_metadata["ContentDate.End"] = pd.to_datetime(
    s3_olci_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

s2_metadata["ContentDate.Start"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s2_metadata["ContentDate.End"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

results = check_collocation(
    s2_metadata, s3_olci_metadata, start_date, end_date,time_window=pd.to_timedelta("10 minutes")
)
```

Display the first 5 results as a table.
```python
from IPython.display import display

display(results.head(5))
```

Display an map visualisation of the first 5 collocated satellite observations. 
```python
from IPython.display import display

map_result = plot_results(results.head(5))
display(map_result)
```


<!-- Proceeding with Sentinel-3 OLCI Download -->
#### Proceeding with Sentinel-3 OLCI Download
This code used to download specific Sentinel-3 OLCI data from the Copernicus database.
```python
download_dir = ""  # Replace with your desired download directory
product_id = results['Satellite1_ID'][0] # Replace with your desired file id
file_name = results['Satellite1_Name'][0]# Replace with your desired filename
# Download the single product
download_single_product(product_id, file_name, access_token, download_dir)
```

#### Sentinel-3 SRAL

This code retrieves and saves Sentinel-3 SRAL metadata.
```python
sentinel3_sral_data = query_sentinel3_sral_arctic_data(
    start_date, end_date, access_token
)

sentinel3_sral_data.to_csv(
    path_to_save_data + "s3_sral_metadata.csv",
    index=False,
)
```
Read the metadata.
```python
s3_sral_metadata = pd.read_csv(
    path_to_save_data + "s3_sral_metadata.csv"
)
s2_metadata = pd.read_csv(
    path_to_save_data + "sentinel2_metadata.csv"
)
```

Process the metadata.
```python
s3_sral_metadata["ContentDate.Start"] = pd.to_datetime(
    s3_sral_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s3_sral_metadata["ContentDate.End"] = pd.to_datetime(
    s3_sral_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

s2_metadata["ContentDate.Start"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["Start"])
).apply(make_timezone_naive)
s2_metadata["ContentDate.End"] = pd.to_datetime(
    s2_metadata["ContentDate"].apply(lambda x: eval(x)["End"])
).apply(make_timezone_naive)

results = check_collocation(
    s2_metadata, s3_sral_metadata, start_date, end_date,time_window=pd.to_timedelta("10 minutes")
)
```

Display an map visualisation of the first 5 collocated satellite observations. 
```python
from IPython.display import display

map_result = plot_results(results.head(5))
display(map_result)
```

