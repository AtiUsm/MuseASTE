# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 03:56:44 2025

@author: Atiya
"""

import re
from html import escape
from datetime import datetime
import time

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import streamlit as st
from annotated_text import annotated_text
from streamlit_extras.mention import mention
from streamlit_echarts import st_echarts
import re
from html import escape
def app():
    
    #   st.set_page_config(
    #    page_title="Car Cmparison Dashboard",
    #    layout="wide",   # or "centered"
    #    initial_sidebar_state="expanded"  # ensures the sidebar opens
    #)
    
    # Dynamically change browser tab title
    #import streamlit as st

    # --- create placeholders at top ---
    col1_placeholder, col2_placeholder = st.columns(2)



    #st.image('car1.jpeg')
    # Titles
    st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
    st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
    mention(
        label="muse-aste-homepage",
        icon="github",
        url="https://github.com/AtiUsm/MuseASTE/tree/main",
    )
    
    st.sidebar.title('Vehicle Comparison Portal')
    x = st.sidebar.feedback("thumbs", key='compf_comp')
    if x == 1:
        st.balloons()

    st.sidebar.pills("Tags", ["ASTE", "Knowledge Graph", "Aspect Sentiment Triplet Extraction","Aspect Based Knowledge Graphs"], key='c_compp')
    st.title("Compare Specifications and Reviews")
    
    
    def construct_triple(aspect, opinion, sentiment):
      return (aspect,opinion,sentiment)
    def construct_triple2(id, topic, aspect):
      return ("car:"+str(id),topicnames[int(topic)],aspect)
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
    
    criterion=st.selectbox('Compare cars on:',options= ['Specifications', 'Phrases','Aspect', 'Sentiment', 'Topic'], key='comparecars_comp')
    
    col1, col2=st.columns(2)
    # Load data
    df = pd.read_csv('objective_ground_truth_complete_perfectum.csv', encoding='latin-1')
        
    with col1:        
        # Sidebar selectbox
        options = df["id"].tolist()
        #default_value = 302
        #default_index = options.index(default_value)
        car_id_1 = st.selectbox(
            "Select Car ID",
            options=options,
            #index=default_index,
            key='car_1_comp',
            index=9
        )
        # Get selected car row
        car_row_1 = df[df['id'] == car_id_1].iloc[0]
    

        st.markdown(f"# Car {car_row_1['id']}: {car_row_1['Make']} {car_row_1['Model']}")

    with col1_placeholder:
        try:
            st.image(f'car_images/24.jpg')
        except:
            st.image('car1.jpeg')    
    with col2:
        car_id_2 = st.selectbox(
            "Select Car ID",
            options=options,
            #index=default_index,
            key='car_2_comp',
            index=4
        )
        
        car_row_2 = df[df['id'] == car_id_2].iloc[0]
        st.markdown(f"# Car {car_row_2['id']}: {car_row_2['Make']} {car_row_2['Model']}")
        # --- later in the code, assign the images ---
    with col2_placeholder:
        try:
            st.image(f'car_images/19.jpg')
        except:
            st.image('car1.jpeg')
    
    
    if criterion=='Specifications':
        with col1:
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
            st.markdown(f"""<p style="font-size:30px;"><b style="color:black;">Objective Parameters:</b></p>""", unsafe_allow_html=True)
            for col in df.columns:
                if col.lower() == "id" or "_" in col:
                    continue
                value = car_row_1[col]
                if str(value).strip().lower() == "not mentioned":
                    continue
                if col.lower() == "color" and ";" in str(value):
                    value = str(value).split(";")[-1].strip()
                if isinstance(value, str) and len(value) > 20:
                    chunks = [value[i:i+20] for i in range(0, len(value), 20)]
                    value = "\n".join(chunks)
                st.markdown(f"""<p style="font-size:20px;"><b style="color:black;">{col}:</b> <span style="color:blue;">{value}</span></p>""", unsafe_allow_html=True)
        with col2:
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
            st.markdown(f"""<p style="font-size:30px;"><b style="color:black;">Objective Parameters:</b></p>""", unsafe_allow_html=True)
            for col in df.columns:
                if col.lower() == "id" or "_" in col:
                    continue
                value = car_row_2[col]
                if str(value).strip().lower() == "not mentioned":
                    continue
                if col.lower() == "color" and ";" in str(value):
                    value = str(value).split(";")[-1].strip()
                if isinstance(value, str) and len(value) > 20:
                    chunks = [value[i:i+20] for i in range(0, len(value), 20)]
                    value = "\n".join(chunks)
                st.markdown(f"""<p style="font-size:20px;"><b style="color:black;">{col}:</b> <span style="color:blue;">{value}</span></p>""", unsafe_allow_html=True)
    if criterion== 'Topic':
        
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
        
        topicn=st.selectbox('Choose Topic', topicnames, key='comTop_comp', index=3)
        
        fields=["id", "label_topic", "aspect","opinion","sentiment"]
        df=pd.read_csv("datasetannotations.csv", usecols=fields) #give the link to train file annotations
        #df2=pd.read_csv("devel_l (1).csv", usecols=fields) #give the link to devel file annotations
        #df=pd.concat([df1,df2],axis=0)
        # df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        with col1:
           triples=get_triples(topicnames.index(topicn),df,car_id_1)
           #st.write(len(triples))
           st.header(topicn)
           if len(triples)==0:
               st.write('No triples in this topic')
           
           p=len([t[2] for t in triples if t[2]=='pos'])
           n=len([t[2] for t in triples if t[2]=='neg'])
           #sentiment_distribution(p,n,key=topicn)
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
           
        with col2:
           triples=get_triples(topicnames.index(topicn),df,car_id_2)
           #st.write(len(triples))
           st.header(topicn)
           if len(triples)==0:
               st.write('No triples in this topic')
           #st.header(topicn)
           p=len([t[2] for t in triples if t[2]=='pos'])
           n=len([t[2] for t in triples if t[2]=='neg'])
           #sentiment_distribution(p,n,key=topicn)
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
    if criterion=='Sentiment':
        fields=["id", "label_topic", "aspect","opinion","sentiment"]
        df=pd.read_csv("dataset_annotations.csv", usecols=fields) #give the link to train file annotations
        #df2=pd.read_csv("devel_l (1).csv", usecols=fields) #give the link to devel file annotations
        #df=pd.concat([df1,df2],axis=0)
        # df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        
            
        
        with col1:
            p1=0
            n1=0
            for topic in range(0,10,1):
                triples=get_triples(topic,df, st.session_state["car_1_comp"],sentiment='pos')
                p1=p1+len(triples)
                triples=get_triples(topic,df,st.session_state["car_1_comp"],sentiment='neg')
                n1=n1+len(triples)
            
            st.header('Total number of positive and negative triples in an entity') 
            sentiment_distribution(p1,n1, key='sentallarrange1_comp')
            
            st.header('Negatives')
        with col2:
            p2=0
            n2=0
            for topic in range(0,10,1):
                triples=get_triples(topic,df,st.session_state["car_2_comp"],sentiment='pos')
                p2=p2+len(triples)
                triples=get_triples(topic,df,st.session_state["car_2_comp"],sentiment='neg')
                n2=n2+len(triples)
            
            st.header('Total number of positive and negative triples in an entity') 
            sentiment_distribution(p2,n2, key='sentallarrange2_comp')
            st.header('Negatives')
    
        
        #container1=col1.container()
        #p=0
        #n=0
        
        with col1:
            for topic in range(0,10,1):
                #st.header('Negatives')
                triples=get_triples(topic,df,st.session_state["car_1_comp"],sentiment='neg')
                #p=p+len(triples)
                if len(triples)==0:
                    continue
                #col1.subheader(topicnames[topic])
                for t in triples:
                        annotated_text(
        #("annotated", "adj", "#faa"),
        (t[1], t[0], "#faa"),
    )
                       
        with col2:
            for topic in range(0,10,1):
                #st.header('Negatives')
                triples=get_triples(topic,df,st.session_state["car_2_comp"],sentiment='neg')
                #n=n+len(triples)
                if len(triples)==0:
                    continue
                #col2.subheader(topicnames[topic])
                for t in triples:
                        annotated_text(
        #("annotated", "adj", "#faa"),
        (t[1], t[0], "#faa"),
    )
                
    
    if criterion == 'Phrases':
        # Search box
        search_term = st.text_input("🔍 Search in reviews (case-insensitive):")
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
        with col1:
            reviews_df = pd.read_csv("car_reviews.csv", encoding='latin-1')   # must have columns: id and text/review
            st.subheader("📋 Review Opinion")
            
            st.markdown("You can choose a different car for the side bar")


            car_reviews = reviews_df[reviews_df["id"] == car_id_1]
            st.caption(f"Showing {len(car_reviews)} reviews for car ID {car_id_1}")

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
                            preview = escape('No matches found.')
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


        # ---------------------------
        # Replace your inner loop with code like this:
        # ---------------------------
        with col2:
            reviews_df = pd.read_csv("car_reviews.csv", encoding='latin-1')   # must have columns: id and text/review
            st.subheader("📋 Review Opinion")
            
            st.markdown("You can choose a different car for the side bar")

            

            # Search box
            #search_term = st.text_input("🔍 Search in reviews (case-insensitive):",key='text2')

            car_reviews = reviews_df[reviews_df["id"] == car_id_2]
            st.caption(f"Showing {len(car_reviews)} reviews for car ID {car_id_2}")

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
                            preview = escape('No matches found.')
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
    if criterion=='Aspect':
        fields=["id", "label_topic", "aspect","opinion","sentiment"]
        df=pd.read_csv("dataset_annotations.csv", usecols=fields) #give the link to train file annotations
        #df2=pd.read_csv("devel_l (1).csv", usecols=fields) #give the link to devel file annotations
        #df=pd.concat([df1,df2],axis=0)
        # df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        df['triple'] = df.apply(lambda x: construct_triple(x.aspect, x.opinion,x.sentiment), axis=1)
        # Search bar
        search_term = st.text_input("Search for an aspect:", key='aspect1_comp')
        with col1:
            #st.title('CAR36')

            id = car_id_1
            aspects = list(set(df.loc[df['id'] == car_id_1, 'aspect']))
            
            if "-" in aspects:
                aspects.remove("-")
            st.title(f"Total Number of Aspects: {len(aspects)}")

            # Dropdown search
            selected_aspect = st.selectbox("Select an aspect:",aspects,key='select_box1_comp')

            


            # Determine which sentiments to include
            selected_sentiments = ['pos', 'neg', 'neu']
            
            # Filter aspects based on search and dropdown
            filtered_aspects = aspects

            if search_term:
                filtered_aspects = [a for a in aspects if search_term.lower() in a.lower()]
            elif selected_aspect != "All":
                filtered_aspects = [selected_aspect]
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
                pass
                #st.write("No aspects match the selected sentiment/search.")

            # Display grid dynamically
            num_columns = 1
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
        with col2:
            #st.title('CAR36')

            id = car_id_2
            aspects = list(set(df.loc[df['id'] == car_id_2, 'aspect']))
            
            if "-" in aspects:
                aspects.remove("-")
            st.title(f"Total Number of Aspects: {len(aspects)}")

            # Dropdown search
            selected_aspect = st.selectbox("Select an aspect:",aspects,key='select_box2_comp')

            # Search bar
            #search_term = st.text_input("Search for an aspect:", key='aspect2')


            # Determine which sentiments to include
            selected_sentiments = ['pos', 'neg', 'neu']
            
            # Filter aspects based on search and dropdown
            filtered_aspects = aspects

            if search_term:
                filtered_aspects = [a for a in aspects if search_term.lower() in a.lower()]
            elif selected_aspect != "All":
                filtered_aspects = [selected_aspect]
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
                pass
                #st.write("No aspects match the selected sentiment/search.")

            # Display grid dynamically
            num_columns = 1
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
#app()