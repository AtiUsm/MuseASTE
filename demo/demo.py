import streamlit as st
from annotated_text import annotated_text
#from st_annotated_text import annotated_text
st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
st.markdown("# Car A")
st.sidebar.markdown("# Profile")
expand=st.sidebar.expander('Features', expanded=True)
#expand.subheader("Objective Properties!")
with expand:
    elements=st.container()
    elements.write('Color: :red[Orange]')
    elements.write('Model: :blue[Lamborgini]')
    elements.write('Type: :green[Convertible]')
    elements.write('Cost: :gray[$300,000]')
    elements.write('Fuel: :gray[Petrol]')



import streamlit as st
import time
st.image("./demo/car1.jpeg")
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
#from streamlit_extras.tags import tagger_component
#import streamlit_toggle_switch as st_toggle_switch
from streamlit_echarts import st_echarts
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
    st_echarts(
        options=option, height="200px", key=key
    )


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
tab1, tab2 , tab3= st.tabs(["Arrange by Topic", "Arrange by Sentiment","View Complete Knowledge Graph"])
#f=st.segmented_control("Filter", ["Open", "Closed"])
#tab1.write(f)
#tab1.write("'Getting Car Summary...'")
#tab2.write("this is tab 2")
tab3.title("Select the maximum no. of ASTE triples per topic")

fields=["id", "segment_id", "label_topic", "aspect","opinion","sentiment"]
df=pd.read_csv("./demo/example_demo.csv", usecols=fields) #give the link to train file annotations
#df2=pd.read_csv("devel_l (1).csv", usecols=fields) #give the link to devel file annotations
#df=pd.concat([df1,df2],axis=0)
df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
#df['triple2'] = df.apply(lambda x: construct_triple2(x.id, x.label_topic,x.aspect), axis=1)

# creates, draws, and saves topic graph. the graph is saved as .graphml file, and images as .pdf
def draw_topic_graph(topic,df,id, a='all', s='all',flag=0):
  if flag==0:  
      triples=get_triples(topicnames.index(topic),df,id,a,s,flag)
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
  nx.write_graphml(G, str(id)+'.graphml')#you can open in other interactive tools
  #save image
  plt.savefig(str(id)+'.pdf') #saving image as pdf for clarity .jpg, .png etc. can be used
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
  plt.savefig(str(id)+'.png')
  nx.write_graphml(G, "CAR:"+str(id)+'.graphml')
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
  plt.savefig(str(id)+'.png')
  nx.write_graphml(G, "CAR:"+str(id)+'.graphml')
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
          time.sleep(0.05)
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
            st.write(sentiment_distribution(p,n,key=topicn))
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
        radio=st.radio('',['pos','neg'], index=0, key=6)
        st.header(':rainbow[Getting all sentimental triples]')
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
          # Update the progress bar with each iteration.
          latest_iteration.text(f'{i+1}')
          bar.progress(i + 1)
          time.sleep(0.05)
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
        for topic in range(0,10,1):
            with col1:
                
                triples=get_triples(topic,df,36,sentiment='pos')
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
                if len(triples)==0:
                    continue
                col2.subheader(topicnames[topic])
                for t in triples:
                        annotated_text(
        #("annotated", "adj", "#faa"),
        (t[1], t[0], "#faa"),
    )
                col2.divider()
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
      time.sleep(0.05)
    '..rendering....'
    #'...and now we\'re done!'
    st.pyplot(entity_graph(36,df,density)) #replace the entiyid by any entity you want
