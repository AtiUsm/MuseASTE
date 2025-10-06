# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:44:17 2025

@author: Atiya
"""


#pip install streamlit
#pip install st-annotated-text
from annotated_text import annotated_text
import streamlit as st
from streamlit_extras.mention import mention
import streamlit as st
import time
st.image("car1.jpeg")
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
#from streamlit_extras.tags import tagger_component
#import streamlit_toggle_switch as st_toggle_switch
from streamlit_echarts import st_echarts
import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:44:17 2025

@author: Atiya
"""

def app():
    
    st.set_page_config(
        page_title="MuSe-CarASTE Objective Features",
        layout="wide",   # or "centered"
        initial_sidebar_state="expanded"  # ensures the sidebar opens
    )

    def plot_car_feature_graph(car_row):
        G = nx.Graph()  
        soft_blue = "#1f77b4"
        car_id = car_row['id']
        center_node = f"Car_{car_id}"
        G.add_node(center_node, label=f"Car: {car_id}", color=soft_blue, edgecolor=soft_blue)
        
        for i, col in enumerate(car_row.index):
            if col.lower() == "id" or "_" in col:
                continue
            value = str(car_row[col]).strip()
            if value.lower() == "not mentioned":
                continue  # skip this column
            col_node = f"{col}_{i}"
            val_node = f"{col}_{i}_val"
            col_label = col
            val_label = str(car_row[col])

            G.add_node(col_node, label=col_label, color=soft_blue, edgecolor=soft_blue)
            G.add_node(val_node, label=val_label, color=soft_blue, edgecolor=soft_blue)
            G.add_edge(center_node, col_node, color='black')
            G.add_edge(col_node, val_node, color='black')

        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_borders = [G.nodes[n]['edgecolor'] for n in G.nodes()]
        edge_colors = [G.edges[e]['color'] for e in G.edges()]

        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(20, 15), dpi=200)
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors=node_borders, node_size=3000, linewidths=2, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, bbox=label_options, font_size=12, font_weight='bold', ax=ax)

        ax.set_title(f"Car Feature Graph: {car_id}", fontsize=40)
        ax.axis('off')

        st.pyplot(fig)

    # Load data
    df = pd.read_csv('objective_ground_truth_complete_perfectum.csv', encoding='latin-1')

    # Sidebar selectbox
    options = df["id"].tolist()
    default_value = 302
    default_index = options.index(default_value)
    car_id = st.sidebar.selectbox(
        "Select Car ID",
        options=options,
        index=default_index
    )

    # Titles
    st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
    st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
    mention(
        label="muse-aste-homepage",
        icon="github",
        url="https://github.com/AtiUsm/MuseASTE/tree/main",
    )

    # Get selected car row
    car_row = df[df['id'] == car_id].iloc[0]

    # Two columns: title & image
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"# Car {car_row['id']}: {car_row['Make']} {car_row['Model']}")
    with col2:
        if car_row['Make'] == 'BMW':
            image_url = "bmw_logo.jpg"; caption = "BMW"
        elif car_row['Make'] == 'Mercedes-Benz':
            image_url = "mercedes_logo.jpg"; caption = "Mercedes"
        elif car_row['Make'] == 'Audi':
            image_url = "audi_logo.jpg"; caption = "Audi"
        else:
            image_url = None; caption = None
        if image_url is not None:
            st.image(image_url, caption=caption, width=80)

    x = st.sidebar.feedback("thumbs")
    if x == 1:
        st.balloons()

    st.sidebar.pills("Tags", ["ASTE", "Knowledge Graph", "Aspect Sentiment Triplet Extraction","Aspect Based Knowledge Graphs"])

    # Tabs
    tabs = st.tabs(['Objective Features'])
    tab1 = tabs[0]

    with tab1:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
            st.markdown(f"""<p style="font-size:30px;"><b style="color:black;">Objective Parameters:</b></p>""", unsafe_allow_html=True)
            for col in df.columns:
                if col.lower() == "id" or "_" in col:
                    continue
                value = car_row[col]
                if str(value).strip().lower() == "not mentioned":
                    continue
                if col.lower() == "color" and ";" in str(value):
                    value = str(value).split(";")[-1].strip()
                if isinstance(value, str) and len(value) > 20:
                    chunks = [value[i:i+20] for i in range(0, len(value), 20)]
                    value = "\n".join(chunks)
                st.markdown(f"""<p style="font-size:20px;"><b style="color:black;">{col}:</b> <span style="color:blue;">{value}</span></p>""", unsafe_allow_html=True)

        with col2:
            plot_car_feature_graph(car_row)

        st.markdown("<br><br>", unsafe_allow_html=True)

        feature_list = []
        for col in df.columns:
            if col.lower() == "id" or "_" in col:
                continue
            value = car_row[col]
            if col.lower() == "color" and ";" in str(value):
                value = str(value).split(";")[-1].strip()
            if isinstance(value, str) and len(value) > 20:
                chunks = [value[i:i+20] for i in range(0, len(value), 20)]
                value = "\n".join(chunks)
            feature_list.append((col, value))

        display_df = pd.DataFrame({col: [val] for col, val in feature_list})
        styled_df = display_df.style.set_properties(**{
            'color': 'blue',
            'font-size': '16px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('color', 'black'), ('font-weight', 'bold'), ('font-size', '16px')]}
        ])
        st.dataframe(styled_df)
    
    
    



