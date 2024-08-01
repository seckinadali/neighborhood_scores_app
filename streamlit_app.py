# running this code in Streamlit: 
# open new terminal
# type "cd src" 
# type "streamlit run Plotting_Streamlit_5.py"

import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Plotting_Streamlit_HELPER_functions import (
    load_data, 
    extract_address_from_path,
    # get_facility_data, # not used here!
    # get_isochrone_data,
    # get_population_data,
    # get_neighborhood_data,
    create_base_map,
    add_original_address,
    add_places,
    add_isochrone,
    add_population,
    plot_facility_counts,
    # convert_address_to_coordinates,
    get_travel_times_Foot_Bike_Car,
    get_travel_time_PT,
    convert_ddhhmmss_to_minutes)

from assign_scores import (assign_custom_scores, assign_cluster_scores)

st.set_page_config(page_title="Comparis Neighborhood Vibe Score", layout="wide") # this is the title appearing in the browser's tab
st.title('The Comparis Neighborhood Vibe Score')
st.markdown("<br>", unsafe_allow_html=True)  # add more "<br>" for more spacing. No effect without ", unsafe_allow_html=True"!

# some global values:
TIME_BOUND = 10 #minutes , also defined in "Plotting_Streamlit_HELPER_functions"

directory = "data/google_data_isochrone_pop_cgpt"


# facility_types = [   # NOT USED ??
#     "bars",
#     "restaurants",
#     "kindergarten",
#     "public_transportation",
#     "gym_fitness",
#     "grocery_stores_supermarkets",
#     "gas_ev_charging",
#     "schools"
# ]

# Initiliazes session_state to avoid that requests are sent to ORS and the PT API each time a widget (like a slider) is moved.
# the functions:
# - get_travel_times_Foot_Bike_Car(work_address, start_coords)
# - get_travel_time_PT(work_address, start_coords)
# are EXECUTED as soon as the max_commute_time is set to NOT NULL
if 'max_commute_time_previous' not in st.session_state:
    st.session_state.max_commute_time_previous = None
# but are ALWAYS executed when the one of the variables property_id OR work_address change
if 'property_id_previous' not in st.session_state:
    st.session_state.property_id_previous = 7 
if 'work_address_previous' not in st.session_state:
    st.session_state.work_address_previous = "Bremgartnerstrasse 51, 8003 Zürich"
# however, if the functions named above are NOT executed, the previous output should be kept
if 'travel_times_df_previous' not in st.session_state:
    st.session_state.travel_times_df_previous = None
if 'commute_time_score_previous' not in st.session_state:
    st.session_state.commute_time_score_previous = None  
# and changing the preferred mode of transport should trigger only the commute_time_score calculation
if 'preferred_mode_transport_previous' not in st.session_state:
    st.session_state.preferred_mode_transport_previous = None  

# SOURCE DATA:
# define files to be used : aim is to display directly the "address" present in the file name in the choice option.
# ideally we have a function which lists all the files in a given folder, extract the ID (Ex1, Ex2) and 
# the relevant address, and updates automatically the "property_map" dictionnary as well as the 
# list in the radio button. 
######################################################################################################

def get_file_names(directory):
    return [
        os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')
    ]

FILES = get_file_names(directory)


######################################################################################################
######################################################################################################
######################################################################################################
# SIDEBAR:
######################################################################################################
st.sidebar.subheader('Personal interests')
######################################################################################################
weights = {
    'Balanced': {
        'gym_fitness': 5,
        'public_transportation': 5,
        'schools': 5,
        'kindergarten': 5,
        'bars': 5,
        'restaurants': 5,
        'gas_ev_charging': 5,
        'grocery_stores_supermarkets': 5
    },
    'Infrastructure': {
        'gym_fitness': 3,
        'public_transportation': 8,
        'schools': 7,
        'kindergarten': 6,
        'bars': 2,
        'restaurants': 4,
        'gas_ev_charging': 6,
        'grocery_stores_supermarkets': 8
    },
    'Entertainment': {
        'gym_fitness': 4,
        'public_transportation': 7,
        'schools': 1,
        'kindergarten': 1,
        'bars': 10,
        'restaurants': 10,
        'gas_ev_charging': 6,
        'grocery_stores_supermarkets': 3
    },
    'Services': {
        'gym_fitness': 5,
        'public_transportation': 8,
        'schools': 7,
        'kindergarten': 6,
        'bars': 2,
        'restaurants': 4,
        'gas_ev_charging': 6,
        'grocery_stores_supermarkets': 8
    }
}
# Radiobutton to choose a weights preset:
weights_preset_id = st.sidebar.selectbox('Following facilities are very (10) or not (0) important to you. Choose one of the presets in the drop-down list and adapt as wished.', list(weights.keys()), key='weights_preset_id')

#  initializes the custom weights:
weights['Custom'] = {}
# Creates a new key in the weight dictionnary and AT THE SAME TIME MAKES 8 SLIDERS OUT OF THEM; 
# FOR THE SLIDERS THE DEFAULT VALUES ("value" argument below) ARE = THE CHOSEN PRESET.
# here potential link to make the sliders green like Comparis: https://discuss.streamlit.io/t/how-to-change-st-sidebar-slider-default-color/3900/4
default_values = weights[weights_preset_id]
for key in list(default_values.keys()): 
    weights['Custom'][key] = st.sidebar.slider(key, min_value=0, max_value=10, value=default_values[key])
# for the scoring the sum of the weights need to add up to 1:
weights_normalized = {}
for key, sub_dict in list(weights.items()):
    total = sum(sub_dict.values())
    weights_normalized[key]  = { key: value / total   for key, value in weights[key].items()}

# create also the normalized weights for the 3 preset weights:
infrastructure_weights_normalized = {key: value  /  sum(weights['Infrastructure'].values())  for key, value in weights['Infrastructure'].items()}
services_weights_normalized = {key: value  /  sum(weights['Services'].values())  for key, value in weights['Services'].items()}
entertainement_weights_normalized = {key: value  /  sum(weights['Entertainment'].values())  for key, value in weights['Entertainment'].items()}

# # DEBUGGING:
# st.write('custom_weights')
# st.write(weights['Custom'])


# infrastructure_weights = weights['Infrastructure']
# services_weights = weights['Services']
# entertainement_weights = weights['Entertainment']
# st.write('infrastructure, services and entertainment')
# st.write(infrastructure_weights_normalized)
# st.write(services_weights_normalized)
# st.write(entertainement_weights_normalized)

# st.write('all weights ?')
# st.write(weights_normalized)

######################################################################################################
st.sidebar.subheader('Layers')
######################################################################################################
facilities_layer = st.sidebar.checkbox('Facilities', value = True)
isochrones_walking = st.sidebar.checkbox('Walking distances', value = True)
population_layer = st.sidebar.checkbox('Population data points', value = False)


#########################################################################################
# MAIN PAGE:
#########################################################################################
# for the drop-down list to choose which property to display:
# First of all: a dictionary to map property IDs to file paths and addresses
property_map = {}
for i in range(len(FILES)):
    property_map[i+1] = {'PROPERTY': FILES[i],
                        'address': extract_address_from_path(FILES[i])}
# Second:  display drop downlist,not in sidebar but as a header on the main page to avoid repetition:
col111, col222, col333, col444 = st.columns([1,1,2,2])
col222.subheader('Select a property :')
# col222.write('') # sets a space for next line "summary style"
col222.write('')

property_id = col333.selectbox('', list(range(1,len(FILES)+1))    , format_func=lambda x: property_map[x]['address'], label_visibility="collapsed", key='property_id')

# select one property and load data:
selected_property = property_map[property_id]
PROPERTY = load_data(selected_property['PROPERTY'])


######################################################################################################
#  MAP, TEXT and NEIGHBORHOOD VIBE SCORE
######################################################################################################
base_map = create_base_map(PROPERTY, 800, 800, 14)

#  ISOCHRONE LAYER:
if isochrones_walking == True:
    add_isochrone(base_map, PROPERTY)
else:
    pass
# POPULATION LAYER:
if population_layer == True:
    add_population(base_map, PROPERTY)
else:
    pass
# PLACES PLAYER:
if facilities_layer == True:
    add_places(base_map, PROPERTY, 15)
else:
    pass
# ORGINAL ADDRESS IN ANY CASE:
add_original_address(base_map, PROPERTY)

base_map.update_layout( margin=dict(r=0, t=0, l=0, b=0), 
                            showlegend=True,
                            legend = dict(  title=dict(text="Facilities", font=dict(size=17, color='black')), 
                                            yanchor="top", y=0.99, xanchor="left", x=0.01,
                                            font=dict(  size=15, color="black"),
                            bgcolor="white")  )


# use 75% of th width for  the map
colx, coly = st.columns([3,1])
colx.plotly_chart(base_map, use_container_width=True)
# Add summary next to the map
coly.subheader('About the neighborhood')
# this list comes directly from the script "Add_Chat_GPT_description"
style_id = coly.selectbox('', ['neutral without emphasis','Real estate agent', 'Lex Fridman'],label_visibility="collapsed")
# style_id = 'neutral without emphasis'
coly.write(PROPERTY['text_description'][style_id]['text'])

# Neighborhood Vibe Score:
##############################3
coly.subheader('Neighborhood Vibe Score')
# Only a placeholder here, definition of plot is further down.
NeighborhoodVibeScore_placeholder = coly.empty()



#  FACILITY COUNTS + PERSONAL INTERESTS SCORE
######################################################################################################
######################################################################################################
col_a, col_b, col_c = st.columns([9,1,3])
# FACILITY COUNTS
######################################################################################################
col_a.subheader(f"Facilities within a {TIME_BOUND}-minute walk ")
col_a.pyplot(plot_facility_counts(PROPERTY), use_container_width=True)

# PERSONAL INTERESTS SCORE
######################################################################################################
col_c.subheader('Personal interests score')
# Only a placeholder here, definition of plot is further down.
PersonalInterestsScore_placeholder = col_c.empty()







#########################################################################################
# COMMUTE TIME TO WORK
#########################################################################################
# User input:
#########################################################################################
st.subheader('Commute time to work')
col1, col2, col3 = st.columns([1,1,2])
col1.markdown('**Maximum acceptable commute time (minutes):**')
col1.write('')
default_address = "Bremgartnerstrasse 51, 8003 Zürich"
# This selectbox also acts as a check box to allow the user to say if it's important or not:  
max_commute_time = col2.number_input('', min_value=5, max_value=None, value=None, placeholder ='I don\'t care', step=5, label_visibility="collapsed", key="max_commute_time")
col1.markdown('**Address:**')
col1.write('')

work_address = col2.text_input('', value = default_address, label_visibility="collapsed", key='work_address')
col1.markdown('**Preferred mode of transport:**')
preferred_mode_transport = col2.selectbox('', ['Foot','Bicycle', 'Car', 'Public transport'],label_visibility="collapsed", key='preferred_mode_transport')

# # DEBUGGING:
# st.write( {key: st.session_state[key] for key in  sorted(st.session_state.keys())   })

# CALCULATE COMMMUTE TIME SCORE
#########################################################################################
if max_commute_time == None: # in this case do not calculate and return routes. Only use a "dummy" commute_time_score
    commute_time_score = 0 ## this is a temporary fix, it must have a value otherwise the custom score does not get calculated.
else:
    # calculating routes must be done ONLY if variables property_id OR work_address change OR if its the first time that a route is calculated => when the PREVIOUS value of max_commute_time == None
    if (property_id != st.session_state.property_id_previous or
        work_address !=  st.session_state.work_address_previous or
        st.session_state.max_commute_time_previous == None ) : # first time that a route is calculated
        # update values: 
        st.session_state.property_id_previous = property_id
        st.session_state.work_address_previous = work_address
        st.session_state.max_commute_time_previous = max_commute_time
        
        # Compute travel times:
        start_coords = PROPERTY['original_address']['coordinates']
        # by Foot_Bike_Car:
        travel_times = get_travel_times_Foot_Bike_Car(work_address, start_coords) # outputs the dictionnary "travel_times" with three keys "Foot", "Bicycle", 'Car'
        # by Public transport:
        shortest_PT_route = get_travel_time_PT(work_address, start_coords)
        # aggregate results:
        travel_times['Public transport'] = convert_ddhhmmss_to_minutes(shortest_PT_route['duration']) # adds the "'Public transport'" key to the preceding dictionnary
        travel_times_df = pd.DataFrame(list(travel_times.items()), columns=['Mode', 'Time (minutes)'])
        st.session_state.travel_times_df_previous = travel_times_df

        # AND define score:
        # get the relevant time to consider for scoring:
        commute_time_to_consider = int( travel_times_df[travel_times_df['Mode']==preferred_mode_transport]['Time (minutes)'] )
        # assign a 1-4 score + update values
        if commute_time_to_consider  > 2 * max_commute_time:
            commute_time_score = 1
            st.session_state.commute_time_score_previous = commute_time_score
        elif commute_time_to_consider  >  max_commute_time:
            commute_time_score = 2
            st.session_state.commute_time_score_previous = commute_time_score
        elif commute_time_to_consider  <  max_commute_time / 2:
            commute_time_score = 4
            st.session_state.commute_time_score_previous = commute_time_score
        elif commute_time_to_consider  >=  max_commute_time / 2:
            commute_time_score = 3
            st.session_state.commute_time_score_previous = commute_time_score
        else: 
            commute_time_score = 2.5
            # and update values
            st.session_state.commute_time_score_previous = commute_time_score

    else: # work_address and property_id do NOT change, but either max_commute_time OR preferred_mode_tranposrt changes => do not calculATE route but calculate commute_time_score => keep travel_times_df but update commute_time_score
        if  (preferred_mode_transport != st.session_state.preferred_mode_transport_previous or
            max_commute_time != st.session_state.max_commute_time_previous):
            # keep travel_times_df
            travel_times_df = st.session_state.travel_times_df_previous
            # update values preferred_mode_transport and max_commute_time
            st.session_state.preferred_mode_transport_previous = preferred_mode_transport
            st.session_state.max_commute_time_previous = max_commute_time
            # DEFINE THE SCORE: 
            # get the relevant time to consider for scoring:
            commute_time_to_consider = int( travel_times_df[travel_times_df['Mode']==preferred_mode_transport]['Time (minutes)'] )
            # assign a 1-4 score:
            if commute_time_to_consider  > 2 * max_commute_time:
                commute_time_score = 1
                st.session_state.commute_time_score_previous = commute_time_score
            elif commute_time_to_consider  >  max_commute_time:
                commute_time_score = 2
                st.session_state.commute_time_score_previous = commute_time_score
            elif commute_time_to_consider  <  max_commute_time / 2:
                commute_time_score = 4
                st.session_state.commute_time_score_previous = commute_time_score
            elif commute_time_to_consider  >=  max_commute_time / 2:
                commute_time_score = 3
                st.session_state.commute_time_score_previous = commute_time_score
            else: 
                commute_time_score = 2.5
                # and update values
                st.session_state.commute_time_score_previous = commute_time_score

# case where OTHER input widgets are moved: travel_times_df  and commute_time_score must still be available => get the last ones   
if max_commute_time != None:
    
    travel_times_df = st.session_state.travel_times_df_previous
    commute_time_score = st.session_state.commute_time_score_previous
    # The dataframe is only showed if commute times have been computed
    col3.dataframe(travel_times_df,use_container_width=True, hide_index=True)
        
# # DEBUGGING:
# col1.write('')
# col2.write('')
# col1.write('')
# col2.write('')
# col1.write(f'DEBUGGING: Commute time score = {commute_time_score}')

# SCORES CALCULATION 
######################################################################################################
######################################################################################################

# initializes the scores dataframe with teh cluster score, then iterates thourgh the weights_normalized dict to compute the other scores (the 4 presets) calculates custom scores (ALTHOUGH THE FUNCTION IS CALLED "assign_custom_scores" IT CAN BE USED TO COMPUTE ANY SCORES)
scores = assign_cluster_scores(directory)
scores.rename(columns={'cluster_score':'Neighborhood Vibe Score'}, inplace = True)
for category in weights_normalized:
    if category != 'Balanced':
        scores_cat = assign_custom_scores(directory,weights_normalized[category], commute_time_score )
        scores_cat.rename(columns={'custom_score':category}, inplace = True)
        scores = scores.merge(scores_cat, on='address')
        del scores_cat
scores.rename(columns={'Custom':'Personal Interests Score'}, inplace = True)


# # DEBUGGING:
# col3.write('DEBUGGING: Scores Dataframe')
# col3.write(scores)

# PLOT SCORES
######################################################################################################
######################################################################################################
property_scores = scores.iloc[property_id-1][1:] # only the scores, not the addres
# ONLY NEIGHBORHOOD SCORE, TO BE DISPLAYED RIGHT FRON THE MAP:
######################################################################################################
fig, ax = plt.subplots(figsize=(1.5,2))
# Color #66cc00 is "Comparis green"
bar2 = sns.barplot(x=property_scores.index[0:1], y=property_scores.values[0:1], ax = ax, color = '#66cc00') 
bar2.bar_label(bar2.containers[0], fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_ylim(0,100)
ax.set_xticks([]) # empty tick lables otherwise redundant info with subheader.
# put that graph in the place holder:
NeighborhoodVibeScore_placeholder.pyplot(fig, use_container_width=False)


# PERSONAL INTERESTS SCORE PLOT
######################################################################################################
# col_c.subheader('Personal interests score')
fig, ax = plt.subplots(figsize=(1,1.5))
bar3 = sns.barplot(x=property_scores.index[4:5], y=property_scores.values[4:5], ax = ax, color = '#017b4f') # color = dark green from Comparis website
bar3.bar_label(bar3.containers[0], fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_ylim(0,100)
ax.set_xticks([]) # empty tick lables otherwise redundant info with subheader.
# Display the plot in the placeholder: 
PersonalInterestsScore_placeholder.pyplot(fig, use_container_width=False)



# FURTHER INFO / DATA SOURCES:
###################################################################################################
st.divider()
st.subheader('Data sources')
st.markdown('[Isochrones for reachability by foot and travel times](https://openrouteservice.org/dev/#/api-docs/v2/isochrones/{profile}/post)')
st.markdown('[Population data](https://www.geocat.ch/geonetwork/srv/eng/catalog.search#/metadata/4bfbbf20-d90e-4131-8fe2-4c454ad45c16)')
st.markdown('[Travel time with public transports](https://transport.opendata.ch/)')


