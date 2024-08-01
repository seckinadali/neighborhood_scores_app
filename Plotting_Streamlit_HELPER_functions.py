# valid for scripts "Plotting_Streamlit_4.py" and up.


# Dependencies
############################################################################################
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import json
# for travel time to work, Foot, Bicycle, Car:
import openrouteservice
from openrouteservice import client as ors

with open('credentials.json', 'r') as file:
    data = json.load(file)
openrouteservice_client = ors.Client(data['ors_key'])
# # for travel time to work, Public Transports:
import requests
############################################################################################

#  Function to load data
#######################################################
def load_data(path):
    try: 
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"An error as occured: {e}")
############################################################################################


#  Function to extract address from source file path
#######################################################
def extract_address_from_path(path:str):
    return  '_'.join(path.split('Ex')[1].split('_')[1:]).split('.json')[0].replace('_',' ')
############################################################################################


# Function to get facilities data in a useable dataframe:
#######################################################
def get_facility_data(data):
    facility_data = []
    for facility_type in data['facilities'].keys():
        for facility_i in data['facilities'][facility_type]['data']:
            facility_data.append({
                'facility_type': facility_type,
                'name': facility_i['name'],
                'lat': facility_i['location']['lat'],
                'lon': facility_i['location']['lng'],
                'address': facility_i['vicinity'], 
                'rating': facility_i['rating'],
                'num_ratings': facility_i['num_ratings'],
                'url': facility_i['url'],
                'travel_time': facility_i['travel_time']
            })
    df = pd.DataFrame(facility_data)
    return df
############################################################################################


# Function to get isochrone data in a useable dataframe:
#######################################################
def get_isochrone_data(data):
    isochrone_data = []
    for isochrone in range(len(data['isochrone']['features'])):
        for coord in range(len(data['isochrone']['features'][isochrone]['geometry']['coordinates'][0])): #("[0]" at the end is necessary since there are two [] in excess!)
            isochrone_data.append({
                'travel_time': data['isochrone']['features'][isochrone]['properties']['value'],
                'lat': data['isochrone']['features'][isochrone]['geometry']['coordinates'][0][coord][1],
                'lon': data['isochrone']['features'][isochrone]['geometry']['coordinates'][0][coord][0],
            })
    df = pd.DataFrame(isochrone_data)
    return df
############################################################################################


# Function to get population data in a useable dataframe:
#######################################################
def get_population_data(data):
    df = pd.DataFrame(data['population']['STATPOP_squares'])
    return df
############################################################################################


# Function to get facility counts: COULD BE NAMED DIFFERENTLY !!!
#######################################################
def get_neighborhood_data(data):
    total_pop = data['population']['total_pop']   
    
    neighborhood_data = []
    for facility_type in data['facilities'].keys():
        neighborhood_data.append({
            'facility_type': facility_type,
            'count_raw': data['facilities'][facility_type]['count'],
            'count_per_10\'000_inhabitants':  round( data['facilities'][facility_type]['count'] / total_pop * 10000 , 1) 
            })
    df = pd.DataFrame(neighborhood_data)
    return df
############################################################################################


# Function to create an EMPTY BASE map:
#################################################
def create_base_map(FILE, width=1500, height=1500, zoom=15):

    # coords of original address:
    LAT = FILE['original_address']['coordinates'][0]
    LON = FILE['original_address']['coordinates'][1]

    # create base map 
    base_map = go.Figure(go.Scattermapbox())
    
    # Set up the layout for the base map
    base_map.update_layout(
        mapbox_style= "open-street-map",
        mapbox_zoom=zoom, 
        mapbox_center={"lat": LAT, "lon": LON},
        width=width,
        height=height,
        margin=dict(r=0, t=0, l=0, b=0), 
        showlegend=False
    )
    return base_map
############################################################################################


# Function to create the layer "Original address"
#################################################
def add_original_address(base_map, FILE):

    # coords of original address:
    LAT = FILE['original_address']['coordinates'][0]
    LON = FILE['original_address']['coordinates'][1]

    # create layer with original adress.
    address_layer = go.Scattermapbox(
        lat=pd.Series(LAT),  
        lon=pd.Series(LON),  
        mode='markers', 
        marker=dict(size=20, color='white', opacity= 1), 
        text = FILE['original_address']['address'],
        hoverinfo='text',
        showlegend=False
    )
    # add the layer on the base map
    base_map.add_trace(address_layer)

    return base_map
############################################################################################


# Function to create the Layer "Places"
############################################
def add_places(base_map, FILE, marker_size=10, ):
    # Color Mapping to be defined by hand and not dynamically so that a bar e.g. is always red
    # better: use icons? See https://www.flaticon.com/search?word=supermarket%20location. 
    # Issues with Scattermapbox: "symbol" argument only reacts to "circle"
    ##############################################################################
    # color_mapping = {
    #     'bars': '#e41a1c',                    # Red
    #     'restaurants': '#377eb8',             # Blue
    #     'kindergarten': '#4daf4a',            # Green
    #     'public_transportation': 'black',   
    #     'gym_fitness': '#ff7f00',             # Orange
    #     'grocery_stores_supermarkets': '#ffcc00', # Yellow dark
    #     'gas_ev_charging': '#a65628',         # Brown
    #     'schools': '#984ea3'                  # Purple
    # }

    #  variations between Dark green and light  green rom Comparis website:
    #  ['#017b4f', '#028b5f', '#03a56f', '#05c07f', '#32cd32', '#4cd964', '#80e075','#66cc00'] 
    color_mapping = {
        'bars': '#017b4f',                    
        'restaurants': '#028b5f',             
        'kindergarten': '#03a56f',            
        'public_transportation': '#05c07f',   
        'gym_fitness': '#32cd32',             
        'grocery_stores_supermarkets': '#4cd964', 
        'gas_ev_charging': '#80e075',         
        'schools': '#66cc00'                 
    }
    # format data
    neighborhood = get_facility_data(FILE)

    # define the text to be dislayed:
    neighborhood['hover_text'] = neighborhood.apply(lambda row: f"<b>{row['name']}</b><br><i>{row['facility_type']}</i><br>{row['address']} <br><br>Average rating: {row['rating']}<br>Nb ratings: {row['num_ratings']}<br><br><i>Walking time: {row['travel_time']}</i> <br><br>", axis=1)

    # add each facility separately on a different trace to be able to show a legend:
    for facility_type, color in color_mapping.items():
        type_data = neighborhood[neighborhood['facility_type'] == facility_type]

        # create the layer:
        places_layer = go.Scattermapbox(
        lat=type_data['lat'],  
        lon=type_data['lon'],  
        mode='markers', 
        marker=dict(size=marker_size,  color=color, opacity= 1), 
        text= type_data['hover_text'],
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        name = facility_type,
        showlegend=True
        )
        # add the layer on the base map
        base_map.add_trace(places_layer)
    
    return base_map
############################################################################################


# Function to display the isochrones
#########################################################################
def add_isochrone(base_map, FILE):
    # format data
    df = get_isochrone_data(FILE)
    # the 6 iscochrones get added on the top of each other so that the closest area is darker
    for travel_time in df['travel_time'].unique():
        df_filtered = df[df['travel_time']==travel_time] 
        # create the layer:
        new_layer = go.Scattermapbox(
            lat=df_filtered['lat'],  
            lon=df_filtered['lon'],  
            mode='lines', # no markers
            line=dict(width=1, color='rgba(10,80,50,1)'),  
            fill='toself', 
            fillcolor='rgba(10,80,50,0.1)', # opcacity = last argument 
            showlegend = False,
            text= str(travel_time/60) + 'min. walking',
            hoverinfo='text'
            )
        base_map.add_trace(new_layer)

    return base_map
############################################################################################


# Function to display the data points of the population counts
#########################################################################
def add_population(base_map, FILE):
    # format data
    df = get_population_data(FILE)
    # create the layer:
    pop_layer = go.Scattermapbox(
    lat=df['E_wgs84'],  
    lon=df['N_wgs84'],  
    mode='markers', 
    marker=dict(size=40,  color='gray', opacity=0.7), 
    text= df['B22BTOT'],
    hoverinfo='text',
    hovertemplate='%{text}<extra></extra>'  ,
    showlegend=False 
    )
    # add the layer on the base map
    base_map.add_trace(pop_layer)

    return base_map
############################################################################################


# Function to display BAR PLOTS of facilities counts
#########################################################################

def plot_facility_counts(GOOGLE_FILE):

    TIME_BOUND = 10 # minutes

    google = get_neighborhood_data(GOOGLE_FILE)
    
    fig, ax = plt.subplots(figsize=(6, 2))

    google_sorted = google#.sort_values('count_raw', ascending=False) # do not sort for better comparison
    
    #  variations between Dark green and light  green rom Comparis website:
    #  ['#017b4f', '#028b5f', '#03a56f', '#05c07f', '#32cd32', '#4cd964', '#80e075','#66cc00'] 
    color_mapping = {
        'bars': '#017b4f',                    
        'restaurants': '#028b5f',             
        'kindergarten': '#03a56f',            
        'public_transportation': '#05c07f',   
        'gym_fitness': '#32cd32',             
        'grocery_stores_supermarkets': '#4cd964', 
        'gas_ev_charging': '#80e075',         
        'schools': '#66cc00'                 
    }
    
    palette =list(color_mapping.values())  # variations between Dark green and light  green rom Comparis website
    # palette = [colors[i % 2] for i in range(len(google_sorted))]
    sns.barplot(data=google_sorted,
                x='count_raw', y=google_sorted['facility_type'], ax=ax, palette = palette )
    ax.set_xlabel('Count', fontsize = 10)
    ax.set_ylabel('', fontsize = 10)
    ax.set_yticklabels(google_sorted['facility_type'], fontsize = 10)
    ax.tick_params(axis='x', labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig
############################################################################################


# Function to get get coordinates from a given address:
######################################################################################################
def convert_address_to_coordinates(address:str):
    """ Transform an address to coordiantes using OpenRouteService
    input: a string representing an address 
    output:  a tuple with the coordinates
    """

    # Initialize client (see beginning of script for the definition of the API key)
    client = openrouteservice_client
    
    # Geocode the address to get its coordinates
    try:
        geocode_result = client.pelias_search(text=address)
        if not geocode_result['features']:
            return {"error": "Address not found"}
        
        address_coords = geocode_result['features'][0]['geometry']['coordinates']
        address_coords = [address_coords[0], address_coords[1]]  # Convert to [lon, lat]
    except openrouteservice.exceptions.ApiError as e:
        return {"error": f"Geocoding error: {e}"}
    
    return address_coords
############################################################################################


# Function to get travel time by FOOT, CYCLING OR CAR, to an hypothetical work address
######################################################################################################
def get_travel_times_Foot_Bike_Car(address_to:str, coordinates_from:tuple):
    """
    Get travel times from a coordinate to an address for walking, cycling and driving using OpenRouteService API.
    :param address: Address, given a s string, to which travel time is calculated
    :param coordinate: Tuple containing the starting latitude and longitude (lat, lon)
    :return: Dictionary with travel times in minutes for walking and driving
    """
    # Initialize client (see beginning of script for the definition of the API key)
    client = openrouteservice_client

    # Define the coordinates in the format required by the API (LON comes first !)
    coords = [[coordinates_from[1], coordinates_from[0]], convert_address_to_coordinates(address_to)]  
    
    # Initialize the result dictionary
    travel_times = {}

    # Get travel time for walking
    try:
        walk_routes = client.directions(coordinates=coords, profile='foot-walking', format='geojson')
        walk_duration = walk_routes['features'][0]['properties']['segments'][0]['duration']/60  # Convert to minutes
        travel_times['Foot'] = int(round(walk_duration,0)) # retain only rounded integer
    except openrouteservice.exceptions.ApiError as e:
        travel_times['walking'] = None
        print(f"Error getting walking directions: {e}")
        
    # Get travel time for bicycle
    try:
        cycling_routes = client.directions(coordinates=coords, profile='cycling-regular', format='geojson')
        cycling_duration = cycling_routes['features'][0]['properties']['segments'][0]['duration']/60  # Convert to minutes
        travel_times['Bicycle'] = int(round(cycling_duration,0)) # retain only rounded integer
    except openrouteservice.exceptions.ApiError as e:
        travel_times['Cycling'] = None
        print(f"Error getting cycling directions: {e}")
    
    # Get travel time for driving
    try:
        car_routes = client.directions(coordinates=coords, profile='driving-car', format='geojson')
        car_duration = car_routes['features'][0]['properties']['segments'][0]['duration']/60  # Convert to minutes
        travel_times['Car'] = int(round(car_duration,0)) # retain only rounded integer
    except openrouteservice.exceptions.ApiError as e:
        travel_times['car'] = None
        print(f"Error getting driving directions: {e}")
    
    return travel_times
############################################################################################


# Function to get travel time by PUBLIC TRANSPORT to an hypothetical work address
######################################################################################################

def get_travel_time_PT(address_from:str, to_coords:tuple):
    """ Uses the transport.opendata.ch API to return the travel time by public transport given an start address and a target coordinate
    - Params: address_from: Start address, will be converted to a tuple of 2 coordinates 
    - Params: to_coords: target coordinates, tuple of 2
    - Returns: a dictionnary with the shortest route found
    """
    from_coords = convert_address_to_coordinates(address_from)

    base_url = "https://transport.opendata.ch/v1/connections"
    
    def get_route(from_coord:tuple, to_coord:tuple):
        params = {
            'from': f"{from_coord[0]},{from_coord[1]}",
            'to': f"{to_coord[0]},{to_coord[1]}"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json().get('connections', [])
        else:
            response.raise_for_status()
    
    # Get routes from API
    routes = get_route(from_coords, to_coords)
    
    if not routes:
        return None

    # Find the best route based on duration
    shortest_route = min(routes, key=lambda x: x['duration'])
        
    return shortest_route
############################################################################################

# Function to convert the output of get_travel_time_PT to minutes:
###################################################################################
def convert_ddhhmmss_to_minutes(duration:str):
    """ Convert a duration in the format dd:hh:mm:ss to total minutes.
    Parameters: duration (str): Duration string in the format dd:hh:mm:ss
    Returns: int: Total minutes
    """
    duration = duration.replace('d', ':')
    parts = duration.split(':')
    if len(parts) != 4:
        raise ValueError("Invalid duration format. Expected format is dd:hh:mm:ss")
    days, hours, minutes, seconds = map(int, parts) # Converts parts to integers
    total_minutes = int( round( days * 24 * 60 + hours * 60 + minutes + seconds // 60 , 0))
    return total_minutes
############################################################################################




