
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
import plotly.express as px
# demo.py
import re
import streamlit as st

st.set_page_config(
    page_title="MuSe-CarASTE-Subjective Features",
    layout="wide",   # or "centered"
    initial_sidebar_state="expanded"  # ensures the sidebar opens
)

dff=pd.read_csv('objective_ground_truth_complete_perfectum.csv', encoding='latin-1')
# Sidebar: Select Car ID dynamically from DataFrame
# Sidebar selectbox
options = dff["id"].tolist()
default_value = 36
default_index = options.index(default_value)
car_id = st.sidebar.selectbox(
    "Select Car ID",
    options=options,
    index=default_index,
    key='car_Select'
)

# Get row corresponding to selected ID
car_row = dff[dff["id"] == car_id].iloc[0]

# Sidebar: Profile
st.sidebar.markdown("# Profile")
expand = st.sidebar.expander("Features", expanded=True)

with expand:
    for col in dff.columns:
        # Skip ID column and columns with underscores
        if col.lower() == "id" or "_" in col:
            continue

        value = car_row[col]
        if value == 'not mentioned':
            continue
        # Assign colors
        if col.lower() == "color":
            color = "gray"
        elif col.lower() == "model":
            color = "gray"
        elif col.lower() == "type":
            color = "gray"
        else:
            color = "gray"

        st.markdown(f"{col}: :{color}[{value}]")

x = st.sidebar.feedback("thumbs")
if x == 1:
    st.balloons()

st.sidebar.pills("Tags", ["ASTE", "Knowledge Graph", "Aspect Sentiment Triplet Extraction","Aspect Based Knowledge Graphs"])


#from st_annotated_text import annotated_text
st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
mention(
        label="muse-aste-homepage",
        icon="github",  # GitHub is also featured!
        url="https://github.com/AtiUsm/MuseASTE/tree/main",
    )

# Get row
car_row = dff[dff["id"] == car_id].iloc[0]
col1, col2 = st.columns([4, 1])  # adjust relative width
with col1:
    # Main dynamic title
    st.markdown(f"# Car {car_row['id']}: {car_row['Make']} {car_row['Model']}")

with col2:
    # Determine image based on Make
    if car_row['Make'] == 'BMW':
        image_url = "bmw_logo.jpg"
        caption = "BMW"
    elif car_row['Make'] == 'Mercedes-Benz':
        image_url = "mercedes_logo.jpg"
        caption = "Mercedes"
    elif car_row['Make'] == 'Audi':
        image_url = "audi_logo.jpg"
        caption = "Audi"
    else:
        image_url = None
        caption = None

    # Display image if available
    if image_url is not None:
        st.image(image_url, caption=caption, width=80)

def sentiment_distribution(p,n,key=0):
    option = {
        "legend": {"top": "bottom"},
        "toolbox": {
            "show": True,
            "feature": {
                "mark": {"show": True},
                "dataView": {"show": True, "readOnly": False},
                "restore": {"show": True},
                "saveAsImage": {"show": True}
            }
        },
        "series": [
            {   "type": "pie",
                "radius": [50, 75],
                "center": ["50%", "50%"],
                "roseType": "area",
                "itemStyle": {"borderRadius": 8},
                "data": [
                    {"value":n, "name": "neg:"+str(n)},
                    {"value":p, "name": "pos:"+str(p)}
                ],
                "color":['red','green']
            }
        ]
    }
    return(st_echarts(
        options=option, height="200px", key=key
    ))


topicnames=['performance',
'interior-features',
'quality-aeshetic',
'comfort',
'handling',
'safety',
'general-information',
'cost',
'user-experience',
'exterior-features']

def construct_triple(aspect, opinion, sentiment):
  return (aspect,opinion,sentiment)
def construct_triple2(id, topic, aspect):
  return ("car:"+str(id),topicnames[int(topic)],aspect)

#extracts positive and negative triples for an entity from all review segments belonging to a particular topic

def get_triples(topic, df,id, aspect='all',sentiment='all', flag=0):
  """
  Input:
    df:dataframe
    topic: topic label (int)
    id: entity id
  Output:
    triples: list of triples
   """
  if aspect=='all' and sentiment=='all':
      subset=df.loc[(df["id"]==id) & (df["sentiment"]!='neu') & (df["sentiment"]!='-') & (df["label_topic"]==topic)]
      #display(subset)
      triples=list(subset["triple"])
  if aspect!='all' and sentiment=='all':
      subset=df.loc[(df["id"]==id) & (df["sentiment"]!='neu') & (df["sentiment"]!='-') & (df["label_topic"]==topic) & (df["aspect"]==aspect)]
      #display(subset)
      triples=list(subset["triple"])
  if sentiment!='all' and aspect=='all':
      subset=df.loc[(df["id"]==id) & (df["sentiment"]==sentiment) &  (df["label_topic"]==topic)]
      #display(subset)
      triples=list(subset["triple"])
  #print(triples)
  if sentiment!='all' and aspect!='all':
      subset=df.loc[(df["id"]==id) & (df["sentiment"]==sentiment) &  (df["label_topic"]==topic) &  (df["aspect"]==aspect)]
      #display(subset)
      triples=list(subset["triple"])
  return triples

# Insert containers separated into tabs:
tab1, tab2 , tab3, tab4, tab5, tab6= st.tabs(["Arrange by Topic", "Arrange by Sentiment","View Subjective Knowledge Graph","Arrange By Aspect", "Review Opinion", "Source and Credibility Metrics"])
#f=st.segmented_control("Filter", ["Open", "Closed"])
#tab1.write(f)
#tab1.write("'Getting Car Summary...'")
#tab2.write("this is tab 2")
tab3.title("Select the maximum no. of ASTE triples per topic")

fields=["id", "segment_id", "label_topic", "aspect","opinion","sentiment"]
df=pd.read_csv("example_demo.csv", usecols=fields) #give the link to train file annotations
#df2=pd.read_csv("devel_l (1).csv", usecols=fields) #give the link to devel file annotations
#df=pd.concat([df1,df2],axis=0)
df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
#df['triple2'] = df.apply(lambda x: construct_triple2(x.id, x.label_topic,x.aspect), axis=1)

# creates, draws, and saves topic graph. the graph is saved as .graphml file, and images as .pdf
def draw_topic_graph(topic,df,id, a='all', s='all',flag=0):
  if flag==0:  
      triples=get_triples(topicnames.index(topic),df,id,a,s,flag)
  if flag==1:
      subset = df.loc[
          (df["id"] == id) &
          (df["aspect"] == a) &
          (df["sentiment"]!='neu') &
          (df["sentiment"]!='-') 
      ]
      triples = list(subset["triple"])
  #aspects=list(set([t[0] for t in triples]))
  st.title(f'{len(triples)} Triples')
  if s=='all' and a=='all':
      pass
      #col1,col2=st.columns(2)
      #col1.title(f'{len(triples)} Triples')
      #p=len([t[2] for t in triples if t[2]=='pos'])
      #n=len([t[2] for t in triples if t[2]=='neg'])
      #st.write(sentiment_distribution(p,n))
  if len(triples)==0:
      st.title('No Triples in this Topic')
      return
  color={'pos':'green','neg':'red'}  
  colormap=[]
  G = nx.DiGraph()
  #G.add_node("car"+str(id),color='pink', edgecolor='black', position='left')
  if a!='all' or s!='all':
    st.write(triples)  
  for aspect, opinion, sentiment in triples:
    if a=='all'and s=='all' or len(list(set([t[0] for t in triples])))>2:  
        G.add_node(topic,color='blue', edgecolor='blue', position='center', label=topic)
    G.add_node(aspect,color='yellow', edgecolor='blue', position='none', label=aspect)
    G.add_node(opinion, color=color.get(sentiment), edgecolor='black', position='none', label=opinion)
    if a=='all' and s=='all' or len(list(set([t[0] for t in triples])))>2:  
        G.add_edge(topic,aspect, color='blue')
    G.add_edge(aspect,opinion, label=sentiment, color=color.get(sentiment))
  #G.add_edge("car"+str(id),topic, color='black')

  nodes=G.nodes()
  edges=G.edges()
  colors = [G.nodes[n]['color'] for n in nodes]
  edgecolors = [G.edges[n]['color'] for n in edges]
  borders=[G.nodes[n]['edgecolor'] for n in nodes]
  # Plot the graph
  fig=plt.figure(figsize=(40,30), dpi=300)

  # Ensures the nodes around the circle are evenly distributed
  pos = nx.kamada_kawai_layout(G)
  #nx.draw(G,pos, node_color=colormap, with_labels=True, font_color='black',node_size=10000,node_shape='8')
  label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
  nx.draw_networkx_nodes(G, pos, node_size=10000,node_color=colors,node_shape='o', edgecolors=borders, linewidths=3)
  nx.draw_networkx_edges(G, pos, edge_color=edgecolors, edgelist=G.edges(), width=3)
  nx.draw_networkx_labels(G, pos, font_size=30, bbox=label_options, font_weight=1.5)
  edge_labels = nx.get_edge_attributes(G, 'label')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=27, font_color='black',rotate=True, bbox=label_options)

  # Display the plot
  plt.suptitle("CAR:"+str(id), fontsize=40)
  plt.axis('off')
  #save graph
  #nx.write_graphml(G, str(id)+'.graphml')#you can open in other interactive tools
  #save image
  #plt.savefig(str(id)+'.pdf') #saving image as pdf for clarity .jpg, .png etc. can be used
  plt.show() 
  return plt
#This creates , draws & saves the entire entity graph listing only positive and negative (not neutral (avoiding it so graph does not becomes big)) across all topics but is too big
def draw_sent_graph(df,id, a='all', sentiment='all',flag=0):
  color={'pos':'green','neg':'red'}
  #id=15.0 #change the entity id here for the enity you want the graph
  colormap=[]
  G = nx.DiGraph()
  G.add_node("car"+str(id),color='pink', edgecolor='black', position='left')
  for topic in range(0,10,1):

    triples=get_triples(topic,df,id,sentiment=sentiment)
    if len(triples)==0:
      continue
    
    topic=topicnames[int(topic)]
    max_nodes=0
    for aspect, opinion, sentiment in triples:
      G.add_node(topic,color='blue', edgecolor='blue', position='center')
      G.add_node(aspect,color='yellow', edgecolor='blue', position='none')
      G.add_node(opinion, color=color.get(sentiment), edgecolor='black', position='none')
      G.add_edge(topic,aspect, color='blue')
      G.add_edge(aspect,opinion, label=sentiment, color=color.get(sentiment))
      max_nodes=max_nodes+1
    G.add_edge("car"+str(id),topic, color='black', edgecolor='black')

  nodes=G.nodes()
  edges=G.edges()
  colors = [G.nodes[n]['color'] for n in nodes]
  edgecolors = [G.edges[n]['color'] for n in edges]
  borders=[G.nodes[n]['edgecolor'] for n in nodes]
  # Plot the graph
  plt.figure(figsize=(25, 25), dpi=300)

  pos = nx.kamada_kawai_layout(G)
  #nx.draw(G,pos, node_color=colormap, with_labels=True, font_color='black',node_size=10000,node_shape='8')
  label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
  nx.draw_networkx_nodes(G, pos, node_size=10000,node_color=colors,node_shape='o', edgecolors=borders, linewidths=3)
  nx.draw_networkx_edges(G, pos, edge_color=edgecolors, edgelist=G.edges(), width=3)
  nx.draw_networkx_labels(G, pos, font_size=20, bbox=label_options)
  edge_labels = nx.get_edge_attributes(G, 'label')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=22, font_color='black',rotate=True, bbox=label_options)

  # Display the plot
  plt.suptitle("CAR:A", fontsize=40)
  plt.axis('off')
  #plt.savefig(str(id)+'.png')
  #nx.write_graphml(G, "CAR:"+str(id)+'.graphml')
  #you can open in other interactive tools. Since the graph is too big save as .graphml, open in other tools
  #st.set_option('deprecation.showPyplotGlobalUse', False)
  plt.show()
  return plt  
def entity_graph(id, df, density):
  color={'pos':'green','neg':'red'}
  #id=15.0 #change the entity id here for the enity you want the graph
  colormap=[]
  G = nx.DiGraph()
  G.add_node("car"+str(id),color='pink', edgecolor='black', position='left')
  for topic in range(0,10,1):

    triples=get_triples(topic,df,id)
    if len(triples)==0:
      continue

    topic=topicnames[int(topic)]
    max_nodes=0
    for aspect, opinion, sentiment in triples:
      G.add_node(topic,color='blue', edgecolor='blue', position='center')
      G.add_node(aspect,color='yellow', edgecolor='blue', position='none')
      G.add_node(opinion, color=color.get(sentiment), edgecolor='black', position='none')
      G.add_edge(topic,aspect, color='blue')
      G.add_edge(aspect,opinion, label=sentiment, color=color.get(sentiment))
      max_nodes=max_nodes+1
      if max_nodes>=density:
          break
    G.add_edge("car"+str(id),topic, color='black', edgecolor='black')

  nodes=G.nodes()
  edges=G.edges()
  colors = [G.nodes[n]['color'] for n in nodes]
  edgecolors = [G.edges[n]['color'] for n in edges]
  borders=[G.nodes[n]['edgecolor'] for n in nodes]
  # Plot the graph
  plt.figure(figsize=(25, 25), dpi=300)

  pos = nx.kamada_kawai_layout(G)
  #nx.draw(G,pos, node_color=colormap, with_labels=True, font_color='black',node_size=10000,node_shape='8')
  label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
  nx.draw_networkx_nodes(G, pos, node_size=10000,node_color=colors,node_shape='o', edgecolors=borders, linewidths=3)
  nx.draw_networkx_edges(G, pos, edge_color=edgecolors, edgelist=G.edges(), width=3)
  nx.draw_networkx_labels(G, pos, font_size=20, bbox=label_options)
  edge_labels = nx.get_edge_attributes(G, 'label')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=22, font_color='black',rotate=True, bbox=label_options)

  # Display the plot
  plt.suptitle("CAR:A", fontsize=40)
  plt.axis('off')
  #plt.savefig(str(id)+'.png')
  #nx.write_graphml(G, "CAR:"+str(id)+'.graphml')
  #you can open in other interactive tools. Since the graph is too big save as .graphml, open in other tools
  #st.set_option('deprecation.showPyplotGlobalUse', False)
  plt.show()
  return plt  
  
   

# You can also use "with" notation:
def compare_entities(id1,id2,topic,df): # function compares any two entities on a particular topic
  triples1=get_triples(topic,df,id1)
  triples2=get_triples(topic,df,id2)
  if len(triples1)!=0:
     draw_topic_graph(triples1,topic,str(id1))
  else:
    print("no triples on this topic "+topicnames[topic]+" for entity "+id1)
  if len(triples2)!=0:
     draw_topic_graph(triples2,topic, str(id2))
  else:
    print("no triples on this topic "+topicnames[topic]+" for entity "+id2)
  print("the two comparitive graphs are plotted and saved")
with tab1:
    f=st.segmented_control("Filter", ["One-by-One", "All"],key=1, default="One-by-One")
    if f=='One-by-One': 
        topic=st.selectbox("Choose Topic", topicnames, key=3)
        'Building topic-wise Knowledge Graph...'
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
          # Update the progress bar with each iteration.
          latest_iteration.text(f'{i+1}')
          bar.progress(i + 1)
          time.sleep(0.001)
        '..now we are done! 3 2 1....'
        #'...and now we\'re done!
        triples=get_triples(topicnames.index(topic),df,36,'all','all')
        p=len([t[2] for t in triples if t[2]=='pos'])
        n=len([t[2] for t in triples if t[2]=='neg'])
        #st.write(sentiment_distribution(p,n))
        st.pyplot(draw_topic_graph(topic,df,36))
        if p>0 and n>0:
            pass
            #st.write(sentiment_distribution(p,n))
        #triples=get_triples(topicnames.index(topic),df,36,'all','all')
        aspects=list(set([t[0] for t in triples]))
        radio=st.radio('Zoom in on aspect',aspects, index=None, key=4)
        if radio!=None:
            st.pyplot(draw_topic_graph(topic,df,36,radio, 'all'))
    if f=='All':
        st.title('Topics')
        st.divider()
        for topicn in topicnames:
            #st.header(topicn)
            triples=get_triples(topicnames.index(topicn),df,36)
            #st.write(len(triples))
            if len(triples)==0:
                continue
            st.header(topicn)
            p=len([t[2] for t in triples if t[2]=='pos'])
            n=len([t[2] for t in triples if t[2]=='neg'])
            sentiment_distribution(p,n,key=topicn)
            for t in triples:
                if t[2]=='pos':
                    annotated_text(
        #("annotated", "adj", "#faa"),
        (t[0], t[1], "#afa"),
    )
                if t[2]=='neg':
                    annotated_text(
        #("annotated", "adj", "#faa"),
        (t[0], t[1], "#faa"),
    )
            st.divider()
#st.segmented_control("Filter", ["Open", "Closed"])    
with tab2:
    f1=st.segmented_control("Filter", ["One-by-One", "All"],key=2, default="One-by-One")
    if f1=='One-by-One': 
        #topic=st.selectbox("Choose Topic", topicnames, key=5)

        #'...and now we\'re done!
        #st.pyplot(draw_topic_graph(topic,df,36))
        #triples=get_triples(topicnames.index(topic),df,36,'all','all')
        #aspects=list(set([t[0] for t in triples]))
        st.title('Choose a sentiment')
        radio=st.radio('which one?',['pos','neg'], index=0, key=6, label_visibility='hidden')
        st.header(':rainbow[Getting all sentimental triples]')
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
          # Update the progress bar with each iteration.
          latest_iteration.text(f'{i+1}')
          bar.progress(i + 1)
          time.sleep(0.001)
        '..now we are done! 3 2 1....'
        st.pyplot(draw_sent_graph(df,36,'all', radio, flag=1))
        topic=st.selectbox("Filter by Topic", topicnames, index=None, key=5)
        if topic:
            st.pyplot(draw_topic_graph(topic,df,36,'all', radio, flag=0))
            #triples=get_triples(topicnames.index(topic),df,36,'all',radio)
            #aspects=list(set([t[0] for t in triples]))
            #radio1=st.radio('Zoom in on aspect',aspects, index=None, key=7)
            #if radio1!=None:
                #st.pyplot(draw_topic_graph(topic,df,36,radio1,radio))
    if f1=='All':
        col1, col2=st.columns(2)
        col1.header('Positives')
        col2.header('Negatives')
        #container1=col1.container()
        p=0
        n=0
        for topic in range(0,10,1):
            with col1:
                
                triples=get_triples(topic,df,36,sentiment='pos')
                p=p+len(triples)
                if len(triples)==0:
                    continue
                col1.subheader(topicnames[topic])
                for t in triples:
                        annotated_text(
        #("annotated", "adj", "#faa"),
        (t[1], t[0], "#afa"),
    )
                col1.divider()        
            with col2:
                
                triples=get_triples(topic,df,36,sentiment='neg')
                n=n+len(triples)
                if len(triples)==0:
                    continue
                col2.subheader(topicnames[topic])
                for t in triples:
                        annotated_text(
        #("annotated", "adj", "#faa"),
        (t[1], t[0], "#faa"),
    )
                col2.divider()
    
    st.header('Total number of positive and negative triples in an entity') 
    sentiment_distribution(p,n, key='sentallarrange')
#st.segmented_control("Filter", ["Open", "Closed"])    
with tab3:
    
    density=st.slider("Graph Density", 1, 10, value=1)
    'Building Knowledge Graph...'
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
      # Update the progress bar with each iteration.
      latest_iteration.text(f'{i+1}')
      bar.progress(i + 1)
      time.sleep(0.001)
    '..rendering....'
    #'...and now we\'re done!'
    st.pyplot(entity_graph(36,df,density)) #replace the entiyid by any entity you want
with tab4:
    f = st.segmented_control("Filter", ["One-by-One", "All"], key="filter_Control_aspect", default="All")

    if f=='One-by-One': 
        #topic=st.selectbox("Choose Apect", aspects, key=3)
        
        
        aspects = list(set(df['aspect']))
        st.title(f"Total Number of Aspects: {len(aspects)}")

        # Dropdown search
        selected_aspect = st.selectbox("Select an aspect:",aspects, key='select_box_1')
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)
        
        'Building topic-wise Knowledge Graph...'
        for i in range(100):
          latest_iteration.text(f'{i+1}')
          bar.progress(i + 1)
          time.sleep(0.001)
        '..now we are done! 3 2 1....'
        
        
        p=len([t[2] for t in triples if t[2]=='pos'])
        n=len([t[2] for t in triples if t[2]=='neg'])
        st.pyplot(draw_topic_graph(topic,df,36,selected_aspect, 'all', flag=1))

    if f == 'All':
        st.title('CAR36')

        id = 36
        aspects = list(set(df['aspect']))
        st.title(f"Total Number of Aspects: {len(aspects)}")

        # Dropdown search
        selected_aspect = st.selectbox("Select an aspect:", ["All"] + aspects,key='select_box2')

        # Search bar
        search_term = st.text_input("Search for an aspect:")

        # Sentiment filter checkboxes
        st.write("Filter by Sentiment:")
        col1, col2, col3 = st.columns(3)
        with col1:
            pos_checked = st.checkbox("Positive", value=True)
        with col2:
            neg_checked = st.checkbox("Negative", value=True)
        with col3:
            neu_checked = st.checkbox("Neutral", value=False)

        # Determine which sentiments to include
        selected_sentiments = []
        if pos_checked:
            selected_sentiments.append('pos')
        if neg_checked:
            selected_sentiments.append('neg')
        if neu_checked:
            selected_sentiments.append('neu')

        # Filter aspects based on search and dropdown
        filtered_aspects = aspects
        if selected_aspect != "All":
            filtered_aspects = [selected_aspect]
        if search_term:
            filtered_aspects = [a for a in filtered_aspects if search_term.lower() in a.lower()]
        if len(filtered_aspects) == 0:
            st.write("No aspects found.")

        # Precompute aspects with at least one matching triple
        display_aspects = []
        aspect_triples_dict = {}
        for aspect in filtered_aspects:
            subset = df.loc[
                (df["id"] == id) &
                (df["aspect"] == aspect) &
                (df["sentiment"].isin(selected_sentiments))
            ]
            triples = list(subset["triple"])
            if len(triples) > 0:
                display_aspects.append(aspect)
                aspect_triples_dict[aspect] = triples

        if len(display_aspects) == 0:
            st.write("No aspects match the selected sentiment/search.")

        # Display grid dynamically
        num_columns = 3
        for i in range(0, len(display_aspects), num_columns):
            row = st.columns(num_columns)
            for j, aspect in enumerate(display_aspects[i:i+num_columns]):
                with row[j]:
                    triples = aspect_triples_dict[aspect]

                    # Build HTML for annotations with flex-wrap
                    html_annotations = "<div style='display:flex; flex-wrap:wrap; gap:4px;'>"
                    for t in triples:
                        if t[2] == 'pos':
                            color = "#afa"  # green
                        elif t[2] == 'neg':
                            color = "#faa"  # red
                        elif t[2] == 'neu':
                            color = "#add8e6"  # blue
                        else:
                            color = "#fff"
                        html_annotations += f"<span style='background-color:{color}; padding:2px 6px; border-radius:3px; white-space:nowrap;'>{t[1]}</span>"
                    html_annotations += "</div>"

                    # Display card with aspect title and annotations inside
                    st.markdown(
                        f"<div style='border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;'>"
                        f"<h4 style='margin:0 0 5px 0'>{aspect}</h4>"
                        f"{html_annotations}</div>",
                        unsafe_allow_html=True
                    )

import re
import streamlit as st
import pandas as pd
from html import escape

# helper: return list of non-overlapping highlighted snippets
def get_merged_highlighted_snippets(text: str, search_term: str, context: int = 300):
    """
    Returns a list of HTML-safe snippets (strings with <mark> highlighting)
    that cover the matches for `search_term` with `context` chars around each match.
    Overlapping windows are merged so no duplicate/repeated snippets are produced.
    """
    if not search_term:
        # no search -> full text (escaped)
        return [escape(text)]

    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []  # caller can decide to show a preview or the full text

    # Build windows around each match
    intervals = []
    for m in matches:
        start = max(m.start() - context, 0)
        end = min(m.end() + context, len(text))
        intervals.append((start, end))

    # Merge overlapping/adjacent intervals
    intervals.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:           # overlaps or touches -> merge
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    snippets = []
    for s, e in merged:
        snippet_raw = text[s:e]

        # Rebuild snippet with proper escaping AND highlighted matches.
        # We escape non-match text and escape matched text inside <mark> to avoid XSS.
        parts = []
        last = 0
        for m in pattern.finditer(snippet_raw):
            # add escaped text before match
            parts.append(escape(snippet_raw[last:m.start()]))
            # add escaped match inside highlight
            parts.append(f"<mark style='background-color: #ffeb3b; color: black;'>{escape(m.group(0))}</mark>")
            last = m.end()
        parts.append(escape(snippet_raw[last:]))  # trailing text

        highlighted_snippet = "".join(parts)

        # Add ellipses if this snippet doesn't include start/end of whole text
        if s > 0:
            highlighted_snippet = "… " + highlighted_snippet
        if e < len(text):
            highlighted_snippet = highlighted_snippet + " …"

        snippets.append(highlighted_snippet)

    return snippets


# ---------------------------
# Replace your inner loop with code like this:
# ---------------------------
with tab5:
    reviews_df = pd.read_csv("car_reviews.csv", encoding='latin-1')   # must have columns: id and text/review
    st.subheader("📋 Review Opinion")

    options = sorted(reviews_df["id"].unique().tolist())
    default_value = 36 if 36 in options else options[0]
    default_index = options.index(default_value)
    car_id = st.selectbox(
        "Select Car ID",
        options=options,
        index=default_index,
        key='car_Select_review'
    )

    # Search box
    search_term = st.text_input("🔍 Search in reviews (case-insensitive):")

    car_reviews = reviews_df[reviews_df["id"] == car_id]
    st.caption(f"Showing {len(car_reviews)} reviews for car ID {car_id}")

    if not car_reviews.empty:
        for idx, row in car_reviews.iterrows():
            # support either 'text' or 'review' column names
            review_text = str(row.get("text", row.get("review", "")))

            if search_term.strip():
                snippets = get_merged_highlighted_snippets(review_text, search_term.strip(), context=300)
                if snippets:
                    # join multiple snippets (top/mid/bottom) with separators
                    display_html = "<br><br> … <br><br>".join(snippets)
                else:
                    # no matches -> show a short gray preview of the top of the review
                    preview = escape(review_text[:500])
                    display_html = f"<span style='color:#888;'>{preview}...</span>"
            else:
                # no search term -> show full escaped review
                display_html = escape(review_text)

            # Render inside a scrollable, styled box
            st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    max-height: 500px;
                    overflow-y: auto;
                    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
                ">
                    <p style="font-size:16px; color:#333; line-height:1.6;">{display_html}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No reviews available for this car.")
with tab6:
        # -------------------------------
        # Example video data (inside Tab 6)
        # -------------------------------
        video_data = {
            "id": "hw8wtDVYtaA",
            "title": "2025 Nissan Armada Review | Consumer Reports",
            "description": "Nissan’s flagship SUV, the 2025 Armada, has been completely redesigned with a bold new look, a twin-turbo V6 engine, upgraded technology, and a more comfortable interior...",
            "comment_count": 65,
            "duration": "16:31",
            "upload_time": "2025-10-03T19:00:07Z",
            "likes": 332,
            "views": 11163,
            "channel_name": "Consumer Reports",
            "channel_subs": 528000,
            "location": "San Francisco, CA",
            "lat": 37.7749,
            "lon": -122.4194,
            "avg_sentiment": 0.11,
            "positive": 41,
            "neutral": 14,
            "negative": 10,
            "num_reviews": 50
        }

        # Top comments (inside Tab 6)
        top_comments = [
            {
                "rank": 1,
                "comment": "I bought a Pro 4X, great to drive and very comfortable. The seats are insanely comfortable!!",
                "likes": 13,
                "time": "2025-10-03T22:58:30Z",
                "author": "@barryhamilton3112"
            },
            {
                "rank": 2,
                "comment": "Name one full-size vehicle where you can sit back in your seat and still be able to reach the dash screen controls… that was a ridiculous critique 😂",
                "likes": 18,
                "time": "2025-10-03T22:05:17Z",
                "author": "@digital_0630"
            }
        ]

        # -------------------------------
        # Streamlit Display
        # -------------------------------
        st.markdown("""
        ### Sample Source and Credibility Metrics
        Sample source video:
        """, unsafe_allow_html=True)
    
        st.video("https://www.youtube.com/embed/hw8wtDVYtaA")
        

        # -------------------------------
        # 📈 Key Metrics
        # -------------------------------
        st.markdown("### 📊 Video Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Views", f"{video_data['views']:,}")
        col2.metric("Likes", f"{video_data['likes']:,}")
        col3.metric("Comments", f"{video_data['comment_count']:,}")

        # -------------------------------
        # 🏛️ Channel & Video Info (Left)
        # -------------------------------
        col_left, col_spacer, col_right = st.columns([1, 0.2, 1])

        with col_left:
            st.markdown("### 🏛️ Channel Credibility")
            st.write(f"**Channel Name:** {video_data['channel_name']}")
            st.write(f"**Number of car reviews done:** {video_data['num_reviews']}")
            st.write(f"**Subscribers:** {video_data['channel_subs']:,}")
            st.write("**Source Reputation:** Verified, independent, nonprofit (Consumer Reports)")

            st.markdown("### 🎥 Video Metadata")
            st.write(f"**Title:** {video_data['title']}")
            st.write(f"**Description:** {video_data['description']}")
            st.write(f"**Duration:** {video_data['duration']}")
            st.write(f"**Upload Time:** {datetime.fromisoformat(video_data['upload_time'].replace('Z','')):%B %d, %Y %H:%M}")
            st.write(f"**Views:** {video_data['views']:,}")
            st.write(f"**Likes:** {video_data['likes']:,}")
            st.write(f"**Comments:** {video_data['comment_count']:,}")
            st.write(f"**Recording Location:** {video_data['location']}")

            if video_data.get("lat") and video_data.get("lon"):
                st.map(pd.DataFrame({"lat": [video_data["lat"]], "lon": [video_data["lon"]]}))
            else:
                st.info("No geolocation coordinates available for this video.")

        # -------------------------------
        # 😊 Sentiment & Comments (Right)
        # -------------------------------
        with col_right:
            st.markdown("### 😊 Audience Sentiment Summary")
            st.write(f"**Average Sentiment Score:** {video_data['avg_sentiment']}")
            st.write(f"- Positive Comments: {video_data['positive']}")
            st.write(f"- Neutral Comments: {video_data['neutral']}")
            st.write(f"- Negative Comments: {video_data['negative']}")

            # Sentiment distribution chart
            sentiment_df = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [video_data["positive"], video_data["neutral"], video_data["negative"]]
            })
            fig = px.pie(sentiment_df, values="Count", names="Sentiment", title="Comment Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # 💬 Top Comments
            st.markdown("### 💬 Top Viewer Comments")
            for c in top_comments:
                st.markdown(f"""
                    <div style='background-color:#f9f9f9;padding:12px;border-radius:10px;margin-bottom:8px'>
                    <b>{c['author']}</b> 
                    <span style='color:gray;font-size:12px'>
                    ({datetime.fromisoformat(c['time'].replace('Z','')):%b %d, %Y %H:%M})</span><br>
                    {c['comment']}<br>
                    <span style='color:#777'>👍 {c['likes']} likes</span>
                    </div>
                """, unsafe_allow_html=True)    
        st.markdown(
            """
            <div style="margin-top: 40px; font-size:14px; color: gray;">
            <br>⚠️ Note: The video used is a sample video from the internet for non-commercial research demo purposes. <br>
            ⚠️ Verify copyright if publishing!<br>
            <br><b>Dataset videos cannot be published publicly. To get authorized access, please sign the End User License Agreement at 
            <a href='https://sites.google.com/view/muse2020/challenge/get-data?authuser=0' target='_blank'>this link</a>.
            The <code>copyright.csv</code> file contains the list of YouTube video links.</b>
            </div>
            """,
            unsafe_allow_html=True
            )
        
