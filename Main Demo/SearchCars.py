# -*- coding: utf-8 -*-
"""
Car Search Portal with advanced aspect-opinion search
"""

import re
from html import escape
import os

import pandas as pd
import streamlit as st


def app():
    st.set_page_config(page_title="Car Search Portal", layout="wide")

    # ---------------------------
    # Helper functions
    # ---------------------------
    STOP_WORDS = {
        "a","an","the","and","or","but","if","in","on","at","for","with","by","to",
        "of","is","are","was","were","be","been","has","have","had","do","does","did",
        "this","that","these","those","as","from","it","its","not"
    }

    def get_words(search_term: str):
        return [w.lower() for w in re.findall(r'\w+', search_term)]

    def highlight_matching_words(text, words, color="#ffeb3b"):
        text_escaped = escape(str(text))
        for word in words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text_escaped = pattern.sub(lambda m: f"<mark style='background-color:{color}'>{m.group(0)}</mark>", text_escaped)
        return text_escaped

    def match_search_in_aspect_opinion(search_words, aspect_text, opinion_text):
        meaningful_words = [w.lower() for w in search_words if w.lower() not in STOP_WORDS]
        if not meaningful_words:
            return False

        aspect_lower = str(aspect_text).lower()
        opinion_lower = str(opinion_text).lower()

        if len(meaningful_words) <= 2:
            aspect_matches = [sw for sw in meaningful_words if sw in aspect_lower]
            opinion_matches = [sw for sw in meaningful_words if sw in opinion_lower and sw not in aspect_matches]
            return bool(aspect_matches) and bool(opinion_matches)
        else:
            combined_text = f"{aspect_text} {opinion_text}".lower()
            return all(sw in combined_text for sw in meaningful_words)

    def get_closest_aspect_opinion_snippet(review_text, aspect, opinion, context=300):
        review_lower = review_text.lower()
        aspect_pos = [m.start() for m in re.finditer(re.escape(aspect.lower()), review_lower)]
        opinion_pos = [m.start() for m in re.finditer(re.escape(opinion.lower()), review_lower)]
        if not aspect_pos or not opinion_pos:
            return None
        min_dist = float("inf")
        best_span = (0, len(review_text))
        for a in aspect_pos:
            for o in opinion_pos:
                dist = abs(a - o)
                if dist < min_dist:
                    min_dist = dist
                    start = max(min(a,o) - context, 0)
                    end = min(max(a,o) + context, len(review_text))
                    best_span = (start, end)
        snippet_raw = review_text[best_span[0]:best_span[1]]
        snippet_html = escape(snippet_raw)
        snippet_html = re.sub(re.escape(aspect), f"<mark style='background-color:#ffeb3b'>{aspect}</mark>", snippet_html, flags=re.IGNORECASE)
        snippet_html = re.sub(re.escape(opinion), f"<mark style='background-color:#90ee90'>{opinion}</mark>", snippet_html, flags=re.IGNORECASE)
        if best_span[0] > 0:
            snippet_html = "… " + snippet_html
        if best_span[1] < len(review_text):
            snippet_html = snippet_html + " …"
        return snippet_html

    # ---------------------------
    # Load data
    # ---------------------------
    obj_df = pd.read_csv("processed_objective.csv", encoding='latin-1')
    reviews_df = pd.read_csv("car_reviews.csv", encoding="latin-1")
    aspects_df = pd.read_csv("dataset_annotations.csv", encoding="latin-1")

    st.title("🚗 Car Search Portal")
    st.caption("Filter cars by objective specs, review text, and aspect–opinion terms")

    # ---------------------------
    # Build search form
    # ---------------------------
    with st.form("car_search_form"):
        st.subheader("⚙️ Objective Filters")

        numeric_cols = [col for col in obj_df.select_dtypes(include="number").columns if "_" not in col and col != "id"]
        categorical_cols = [col for col in obj_df.columns if obj_df[col].dtype == "object" and "_" not in col and col not in ["id", "review"]]

        col1, col2 = st.columns(2)
        numeric_filters = {}
        categorical_filters = {}

        # Numeric filters
        with col1:
            for col in numeric_cols:
                min_val = float(obj_df[col].min())
                max_val = float(obj_df[col].max())
                slider_val = st.slider(f"{col} (Range)", min_val, max_val, (min_val, max_val), key=f"{col}_range")
                input_min_str = st.text_input(f"{col} exact (optional)", value="", key=f"{col}_exact")
                input_min = float(input_min_str) if input_min_str.strip() != "" else None
                if input_min is None:
                    input_min, input_max = slider_val
                else:
                    input_max = input_min
                numeric_filters[col] = (input_min, input_max)

        # Categorical filters
        with col2:
            for col in categorical_cols:
                options = sorted([v for v in obj_df[col].dropna().unique() if str(v).strip().lower() != "not mentioned"])
                categorical_filters[col] = st.multiselect(f"{col} (Optional)", options=options, key=f"{col}_multi")

        st.markdown("---")
        st.subheader("💬 Review & Aspect Search (Optional)")
        search_term = st.text_input("Enter word/phrases (use AND/OR for multiple):", key="review_term")

        submit = st.form_submit_button("🔍 Search Cars")

    # ---------------------------
    # Handle submission
    # ---------------------------
    filtered_df = pd.DataFrame(columns=obj_df.columns)
    if submit:
        filtered_df = obj_df.copy()

        # Apply numeric filters
        for col, (min_val, max_val) in numeric_filters.items():
            if min_val != obj_df[col].min() or max_val != obj_df[col].max():
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        # Apply categorical filters
        for col, selected_vals in categorical_filters.items():
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].apply(lambda x: any(sel.lower() in str(x).lower() for sel in selected_vals))]

        # ---------------------------
        # Parse multiple phrases and logic
        # ---------------------------
        matched_ids_from_aspects = set()
        matched_ids_from_reviews = set()
        if search_term.strip():
            # Parse AND/OR
            phrases = re.split(r"\s+AND\s+|\s+OR\s+", search_term, flags=re.IGNORECASE)
            operator = "AND" if "AND" in search_term.upper() else "OR"
            search_words_per_phrase = [get_words(p) for p in phrases]

            # Aspect matching per phrase
            phrase_matches = []
            for words in search_words_per_phrase:
                matches = aspects_df.apply(lambda row: match_search_in_aspect_opinion(words, row['aspect'], row['opinion']), axis=1)
                phrase_matches.append(aspects_df[matches]['id'].unique())

            # Combine per operator
            if phrase_matches:
                if operator == "AND":
                    matched_ids_from_aspects = set.intersection(*map(set, phrase_matches))
                else:
                    matched_ids_from_aspects = set.union(*map(set, phrase_matches))

            # Review text search
            text_col = "review" if "review" in reviews_df.columns else "text"
            reviews_df["review_text"] = reviews_df[text_col].fillna("")
            matched_ids_from_reviews = reviews_df[reviews_df["review_text"].str.contains("|".join(phrases), case=False, na=False)]["id"].unique()

        # Combine all matched IDs
        if search_term.strip():
            all_matched_ids = set(matched_ids_from_aspects) | set(matched_ids_from_reviews)
            filtered_df = filtered_df[filtered_df["id"].isin(all_matched_ids)]

        st.markdown("---")
        st.success(f"✅ {len(filtered_df)} cars found matching your criteria.")
        if len(filtered_df) == 0:
            st.warning("No cars match your filters. Try broadening your search.")
            return

        # ---------------------------
        # Display matching cars
        # ---------------------------
        image_folder = "car_images"
        for _, row in filtered_df.iterrows():
            car_id = row["id"]
            make = row.get("Make", "") or "Unknown Make"
            model = row.get("Model", "") or "Unknown Model"
            st.markdown(f"### 🚘 {make} {model} (ID: {car_id})")

            # Objective specs
            with st.expander("View Objective Specifications"):
                specs = {col: row[col] for col in numeric_cols + categorical_cols if col in row and pd.notna(row[col])}
                if specs:
                    st.json(specs)
                else:
                    st.info("No valid specifications available.")

            # Review snippets with highlighted aspect & opinion
            if search_term.strip():
                car_reviews = reviews_df[reviews_df["id"] == car_id]
                car_aspects = aspects_df[aspects_df["id"] == car_id]
                for _, review_row in car_reviews.iterrows():
                    review_text = str(review_row.get("review", review_row.get("text", "")))
                    for _, arow in car_aspects.iterrows():
                        for words in search_words_per_phrase:
                            if match_search_in_aspect_opinion(words, arow['aspect'], arow['opinion']):
                                snippet = get_closest_aspect_opinion_snippet(review_text, arow['aspect'], arow['opinion'])
                                if snippet:
                                    st.markdown(f"<div style='background-color:#f8f9fa; padding:10px; margin-bottom:10px'>{snippet}</div>", unsafe_allow_html=True)

        # Summary of matching cars
        st.markdown("---")
        st.subheader("📋 Matching Cars Summary")
        unique_cars = filtered_df.drop_duplicates(subset=['id']).reset_index(drop=True)
        summary_lines = [f"{row['id']} — {row.get('Make','not mentioned')} {row.get('Model','not mentioned')}" for _, row in unique_cars.iterrows()]
        summary_text = "\n".join(summary_lines)
        st.text_area(f"{len(summary_lines)} Cars satisfying your query:", summary_text, height=150, disabled=True)

app()
