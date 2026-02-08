# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:19:50 2026

@author: Atiya
"""
import re
from html import escape
from datetime import datetime
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from streamlit_extras.mention import mention
from streamlit_echarts import st_echarts

def app():
    # -------------------------------
    # Load datasets
    # -------------------------------
    cff = pd.read_csv('objective_ground_truth_complete_perfectum.csv', encoding='latin-1')
    cf = pd.read_csv("full annotations.csv", encoding="latin1") 

    subset = cf[cf["sentiment"] != '-']

    # Construct triples
    def construct_triple(aspect, opinion, sentiment):
        return (aspect, opinion, sentiment)
    cf['triple'] = cf.apply(lambda x: construct_triple(x.aspect, x.opinion, x.sentiment), axis=1)

    # -------------------------------
    # Sidebar
    # -------------------------------
    st.sidebar.title('Insight Portal')
    x = st.sidebar.feedback("thumbs", key='f_ins')
    if x == 1:
        st.balloons()
    st.sidebar.pills("Tags", ["ASTE", "Knowledge Graph", "Aspect Sentiment Triplet Extraction","Aspect Based Knowledge Graphs"], key='p_ins')

    # -------------------------------
    # Titles
    # -------------------------------
    st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
    st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
    mention(label="muse-aste-homepage", icon="github", url="https://github.com/AtiUsm/MuseASTE/tree/main")

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    all_main_categories = sorted(subset["MAIN_CATEGORY"].dropna().unique().tolist())
    selected_main_categories = st.multiselect("Select Main Categories", options=all_main_categories, default=None, key="MAINCATEGORY_ins")

    base_categories = subset["base_category"].dropna().str.split(",").explode().str.strip().unique()
    selected_base_categories = st.multiselect("Select Base Categories", options=base_categories, default=None, key="BASECATEGORY_ins")

    topicnames = ['performance','interior-features','quality-aesthetic','comfort','handling','safety','general-information','cost','user-experience','exterior-features']
    selected_topic = st.multiselect("Select Topics", options=topicnames, default=None, key="topicnames_ins")

    sentiments = ['pos','neu','neg']
    selected_sentiment = st.multiselect("Select Sentiments", options=sentiments, default=None, key="sentiments_ins")

    car_ids = st.multiselect("Select Car ID", options=cff["id"].tolist(), default=None, key='car_Select_ins')

    # -------------------------------
    # Dynamic aspects
    # -------------------------------
    if car_ids:
        aspects = list(set(subset.loc[subset['id'].isin(car_ids), 'aspect']))
    else:
        aspects = list(set(subset['aspect']))
    if "-" in aspects:
        aspects.remove("-")
    selected_aspects = st.multiselect("Select Aspects", options=aspects, default=None, key="aspects_ins")

    # Aspect search
    search_term = st.text_input("Search for an aspect:", key='a1_ins')
    if search_term:
        selected_aspects = [a for a in aspects if search_term.lower() in a.lower()]


    # -------------------------------
    # AND / OR logic selection
    # -------------------------------
    internal_logic = st.radio("Internal logic within each filter (AND/OR):", options=["AND", "OR"], index=1, horizontal=True)
    cross_logic = st.radio("Combine filters across select boxes (AND/OR):", options=["AND", "OR"], index=0, horizontal=True)

    # -------------------------------
    # Ultimate Insight Search Function
    # -------------------------------
    def run_ultimate_insight_search_flexible(
        subset,
        main_categories=None,
        base_categories=None,
        topics=None,
        sentiments=None,
        car_ids=None,
        aspects=None,
        internal_logic="OR",
        cross_logic="AND"
    ):

        def apply_dynamic_filters():
            filtered = subset.copy()
            masks = []

            # Helper for internal logic
            def internal_mask(column, selections):
                if internal_logic == "OR":
                    return filtered[column].isin(selections)
                else:  # AND
                    return filtered[column].apply(lambda val: all(s in str(val).split(",") for s in selections))

            # Masks per filter
            if main_categories:
                masks.append(internal_mask("MAIN_CATEGORY", main_categories))
            if topics:
                masks.append(internal_mask("topic", topics))
            if sentiments:
                masks.append(internal_mask("sentiment", sentiments))
            if car_ids:
                masks.append(filtered["id"].isin(car_ids))
            if aspects:
                masks.append(filtered["aspect"].isin(aspects))
            if base_categories:
                def match_base(row):
                    combined = f"{row['base_category']},{row['EXTRA_Opinion_Base_Needed_RARE_EXAMPLE']}"
                    values = [x.strip() for x in str(combined).split(",")]
                    if internal_logic == "OR":
                        return any(v in base_categories for v in values)
                    else:
                        return all(v in values for v in base_categories)
                masks.append(filtered.apply(match_base, axis=1))

            # Combine across filters
            if masks:
                if cross_logic == "AND":
                    combined_mask = masks[0]
                    for m in masks[1:]:
                        combined_mask = combined_mask & m
                else:  # OR
                    combined_mask = masks[0]
                    for m in masks[1:]:
                        combined_mask = combined_mask | m
                filtered = filtered[combined_mask]

            return filtered

        # Button to trigger filtering
        if st.button("🔍 Get Insight"):
            filtered_df = apply_dynamic_filters()
            st.markdown(f"## 📊 Results: {len(filtered_df)} triples found")
            if filtered_df.empty:
                st.warning("No triples found for current selection.")
                return

            color_map = {"pos": "green", "neg": "red", "neu": "blue"}
            for _, row in filtered_df.iterrows():
                color = color_map.get(row["sentiment"], "black")
                st.markdown(
                    f"- :{color}[**Aspect:** {row['aspect']} | **Opinion:** {row['opinion']} | **Sentiment:** {row['sentiment']}]"
                )

    # -------------------------------
    # Call the search function
    # -------------------------------
    run_ultimate_insight_search_flexible(
        subset=subset,
        main_categories=selected_main_categories,
        base_categories=selected_base_categories,
        topics=selected_topic,
        sentiments=selected_sentiment,
        car_ids=car_ids,
        aspects=selected_aspects,
        internal_logic=internal_logic,
        cross_logic=cross_logic
    )
