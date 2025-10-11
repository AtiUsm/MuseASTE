# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:01:59 2025
@author: Atiya
"""

import re
from html import escape
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from streamlit_extras.mention import mention
import pandas as pd
import streamlit as st



def app():
    # ---------------------------
    # Helper functions
    # ---------------------------
    
    #st.image('car1.jpeg')
    # Titles
    st.title(':blue[MuSe-CarASTE: A comprehensive dataset for aspect sentiment triplet extraction in automotive review videos]')
    st.title(':green[Get a sneak-peek into our dataset! :eyes:]')
    mention(
        label="muse-aste-homepage",
        icon="github",
        url="https://github.com/AtiUsm/MuseASTE/tree/main",
    )
    
    st.sidebar.title('Automobile Finder: Comprehensive Car Search')
    x = st.sidebar.feedback("thumbs", key='sfeed1_search')
    if x == 1:
        st.balloons()

    st.sidebar.pills("Tags", ["ASTE", "Knowledge Graph", "Aspect Sentiment Triplet Extraction","Aspect Based Knowledge Graphs"], key='spill_search')
    
    
    def get_words(search_term: str):
        return [w.lower() for w in re.findall(r'\w+', search_term)]

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "for", "with",
        "by", "to", "of", "is", "are", "was", "were", "be", "been", "has", "have",
        "had", "do", "does", "did", "this", "that", "these", "those", "as", "from",
        "it", "its", "not"
    }

    def highlight_matching_words(text, search_words, color="#ffeb3b"):
        """Escape non-matching text and highlight matches with <mark>."""
        if not text:
            return ""
        raw = str(text)
        if not search_words:
            return escape(raw)
        pattern = re.compile("|".join(re.escape(w) for w in search_words), re.IGNORECASE)

        last = 0
        parts = []
        for m in pattern.finditer(raw):
            parts.append(escape(raw[last:m.start()]))
            parts.append(f"<mark style='background-color:{color}'>{escape(m.group(0))}</mark>")
            last = m.end()
        parts.append(escape(raw[last:]))
        return "".join(parts)
    

    def merge_duplicate_snippets(snippets):
        """
        Merge snippets with the same base text but possibly different highlights.
        Deduplicate exact identical snippets.
        """
        merged_snippets = []
        seen_texts = {}
        
        for sn in snippets:
            # Remove HTML tags to get raw text for comparison
            raw = re.sub(r'<[^>]+>', '', sn)
            if raw in seen_texts:
                # Merge highlights if different
                if sn not in seen_texts[raw]:
                    seen_texts[raw].append(sn)
                else:
                    seen_texts[raw] = [sn]

        # Combine merged snippets
        for raw, variants in seen_texts.items():
            if len(variants) == 1:
                merged_snippets.append(variants[0])
            else:
                # Combine all highlight variants by replacing each with <mark>
                combined = raw
                for variant in variants:
                    marks = re.findall(r'(<mark.*?>.*?</mark>)', variant)
                    for mark in marks:
                        # Replace text in combined with highlighted HTML
                        mark_text = re.sub(r'<[^>]+>', '', mark)
                        combined = combined.replace(mark_text, mark, 1)
                merged_snippets.append(combined)

            return merged_snippets


    def match_search_in_aspect_opinion(search_words, aspect_text, opinion_text):
        """
        Matching logic:
         - Remove stop words
         - If <=2 meaningful words: require at least one match in aspect AND at least one (different) match in opinion
         - If >2 meaningful words: require all meaningful words to appear somewhere in (aspect+opinion)
        """
        meaningful_words = [w.lower() for w in search_words if w.lower() not in STOP_WORDS]
        if not meaningful_words:
            return False

        aspect_lower = str(aspect_text).lower()
        opinion_lower = str(opinion_text).lower()

        if len(meaningful_words) <= 2 and len(meaningful_words)>1:
            aspect_matches = [sw for sw in meaningful_words if sw in aspect_lower]
            opinion_matches = [sw for sw in meaningful_words if sw in opinion_lower and sw not in aspect_matches]
            return bool(aspect_matches) and bool(opinion_matches)
        elif len(meaningful_words) ==1:
            aspect_matches = [sw for sw in meaningful_words if sw in aspect_lower]
            opinion_matches = [sw for sw in meaningful_words if sw in opinion_lower and sw not in aspect_matches]
            return bool(aspect_matches) or bool(opinion_matches)
        else:
            combined_text = f"{aspect_text} {opinion_text}".lower()
            return all(sw in combined_text for sw in meaningful_words)

    def get_closest_aspect_opinion_snippet(review_text, aspect, opinion, context=300):
        """
        Return a safe HTML snippet with aspect/opinion highlighted (aspect yellow, opinion green).
        If either aspect or opinion not found, return None.
        """
        if not review_text:
            return None
        text = str(review_text)
        text_lower = text.lower()
        aspect_l = str(aspect).lower()
        opinion_l = str(opinion).lower()

        aspect_pos = [m.start() for m in re.finditer(re.escape(aspect_l), text_lower)]
        opinion_pos = [m.start() for m in re.finditer(re.escape(opinion_l), text_lower)]
        if not aspect_pos or not opinion_pos:
            return None

        min_dist = float("inf")
        best_span = (0, len(text))
        for a in aspect_pos:
            for o in opinion_pos:
                dist = abs(a - o)
                if dist < min_dist:
                    min_dist = dist
                    start = max(min(a, o) - context, 0)
                    end = min(max(a, o) + context, len(text))
                    best_span = (start, end)

        s, e = best_span
        snippet_raw = text[s:e]
        snippet_lower = snippet_raw.lower()

        # Find matches within snippet and build escaped HTML with marks
        matches = []
        for m in re.finditer(re.escape(aspect_l), snippet_lower):
            matches.append(("aspect", m.start(), m.end()))
        for m in re.finditer(re.escape(opinion_l), snippet_lower):
            matches.append(("opinion", m.start(), m.end()))
        matches.sort(key=lambda x: x[1])

        out_parts = []
        last = 0
        for tag, ms, me in matches:
            out_parts.append(escape(snippet_raw[last:ms]))
            matched_text = snippet_raw[ms:me]
            if tag == "aspect":
                out_parts.append(f"<mark style='background-color:#ffeb3b'>{escape(matched_text)}</mark>")
            else:
                out_parts.append(f"<mark style='background-color:#90ee90'>{escape(matched_text)}</mark>")
            last = me
        out_parts.append(escape(snippet_raw[last:]))

        snippet_html = "".join(out_parts)
        if s > 0:
            snippet_html = "… " + snippet_html
        if e < len(text):
            snippet_html = snippet_html + " …"
        return snippet_html

    def get_merged_highlighted_snippets(text: str, search_term: str, context: int = 300):
        """
        Return snippets centered around occurrences of the exact phrase search_term
        with that phrase highlighted.
        """
        if not search_term:
            return [escape(text)]
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            return []
        intervals = []
        for m in matches:
            start = max(m.start() - context, 0)
            end = min(m.end() + context, len(text))
            intervals.append((start, end))
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        snippets = []
        for s, e in merged:
            snippet_raw = text[s:e]
            snippet_html = escape(snippet_raw)
            snippet_html = pattern.sub(lambda m: f"<mark style='background-color:#ffeb3b'>{escape(m.group(0))}</mark>", snippet_html)
            if s > 0:
                snippet_html = "… " + snippet_html
            if e < len(text):
                snippet_html = snippet_html + " …"
            snippets.append(snippet_html)
        return snippets

    # ---------------------------
    # Load data
    # ---------------------------    @st.cache_data
    def load_data(usecols=None):
        obj_df = pd.read_csv("processed_objective.csv", encoding='latin-1', usecols=usecols)
        reviews_df = pd.read_csv("car_reviews.csv", encoding="latin-1", usecols=None)
        aspects_df = pd.read_csv("dataset_annotations.csv", encoding="latin-1",usecols=None)
        return obj_df, reviews_df, aspects_df

    obj_df, reviews_df, aspects_df = load_data()
    obj_df['id'] = obj_df['id'].astype(str)
    reviews_df['id'] = reviews_df['id'].astype(str)
    aspects_df['id'] = aspects_df['id'].astype(str)
    
    st.title("🚗 Car Search Portal")
    st.caption("Filter cars by objective specs, review text, and aspect–opinion terms")

    # ---------------------------
    # Build the search form
    # ---------------------------
    with st.form("car_search_form"):
        st.subheader("⚙️ Objective Filters")
        # ✅ Numeric and categorical column detection
        numeric_cols = [
            col for col in obj_df.select_dtypes(include="number").columns
            if "_" not in col and col not in'id'
            ]
        categorical_cols = [
            col for col in obj_df.columns
            if obj_df[col].dtype == "object"
            and "_" not in col
            and col.lower() not in ["review", "text"]
            ]
        col1, col2 = st.columns(2)
        numeric_filters = {}
        categorical_filters = {}

        # Numeric filters
        with col1:
            for col in numeric_cols:
                min_val = float(obj_df[col].min())
                max_val = float(obj_df[col].max())
                slider_val = st.slider(f"{col} (Specify Range:)", min_val, max_val, (min_val, max_val), key=f"{col}_range_search")
                col_min_input, _ = st.columns(2)
                with col_min_input:
                    input_min_str = st.text_input(f" Exact Value {col} (optional)", value="", key=f"{col}_min_input_search")
                input_min = float(input_min_str) if input_min_str.strip() != "" else None
                if input_min is None:
                    input_min, input_max = slider_val
                else:
                    input_max = input_min
                numeric_filters[col] = (input_min, input_max)

        # Categorical filters
        with col2:
            for col in categorical_cols:
                if col.lower() == "color":
                    colors = obj_df[col].dropna().astype(str)
                    unique_colors = set()
                    for c in colors:
                        if c.strip().lower() in ["", "not mentioned"]:
                            continue
                        parts = re.split(r',|-', c)
                        for p in parts:
                            p_clean = p.strip().title()
                            if p_clean:
                                unique_colors.add(p_clean)
                    options = sorted(unique_colors)
                else:
                    options = sorted([v for v in obj_df[col].dropna().unique() if str(v).strip().lower() != "not mentioned"])
                categorical_filters[col] = st.multiselect(f"{col} (Optional)", options=options, key=f"{col}_multi_search")

        st.markdown("---")
        st.subheader("💬 Review & Aspect Opinion Search (Optional)")
        search_term = st.text_input("Enter a word or phrase:", key="review_term_search")

        st.subheader("Examples:")
        st.write("large boot")
        st.write("good stereo system")
        st.write("comfortable seats")
        st.write("futuristic design")
        st.write("loads of headroom")
        st.write("soft materials")
        st.write("sharp steering")
        st.write("quick engine")
        st.write("windows big")
        st.write("new technology")
        st.write("large door bins")
        st.write("saves fuel")
        st.write("strong back pockets")
        st.write("easy to use display")
        st.write("flat steering")
        st.write("good grip")
        st.write("corner handling hard")
        st.write("quiet car")
        st.write("family car")
        st.write("smooth gear change")
        st.write("low seating- if you are short! ;) ")
        submit = st.form_submit_button("🔍 Search Cars")
        st.markdown(
           """
           <div style="margin-top: 40px; font-size:14px; color: gray;">
           <br><br><br>⚠️ Note: You can enter multiple phrases with and/or but we recommend the following:
               <br>⚠️ Note: For demo purposes, enter one phrase as shown in the example (not multiple phrases with conjunction).
           <br>⚠️ Note:The search function is very primitive for demo purpose it may contain some false positives.
           <br>⚠️ Note:The images used are automatically crawled from the internet for the sake of demo for non-commercial research purposes. Verify copyright if publishing!
           <br><br><b>Dataset images cannot be published publicly. To get authorized access to the dataset, please sign the End User License Agreement at 
           <a href='https://sites.google.com/view/muse2020/challenge/get-data?authuser=0' target='_blank'>this link</a>.</b>
           </div>
           """,
           unsafe_allow_html=True
       )

    # ---------------------------
    # Handle submission
    # ---------------------------
    #import pandas as pd

    filtered_df = pd.DataFrame(columns=list(obj_df.columns) )

    if submit:
        obj_df, reviews_df, aspects_df = load_data(usecols=['id']+categorical_cols+numeric_cols)
        filtered_df = obj_df.copy()

        # Numeric filters
        for col, (min_val, max_val) in numeric_filters.items():
            if min_val != obj_df[col].min() or max_val != obj_df[col].max() and col in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        # Categorical filters
        for col, selected_vals in categorical_filters.items():
            if selected_vals and col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].apply(
                    lambda x: any(sel.lower() in str(x).lower() for sel in selected_vals)
                )]

        # ---------------------------
        # Parse phrases & search logic
        # ---------------------------
        all_matched_ids = set()
        phrases = []
        logic_mode = "SINGLE"
        if search_term.strip():
            raw_term = search_term.strip()
            if re.search(r"\band\b", raw_term, flags=re.IGNORECASE):
                logic_mode = "AND"
                phrases = [p.strip() for p in re.split(r"\band\b", raw_term, flags=re.IGNORECASE) if p.strip()]
            elif re.search(r"\bor\b", raw_term, flags=re.IGNORECASE):
                logic_mode = "OR"
                phrases = [p.strip() for p in re.split(r"\bor\b", raw_term, flags=re.IGNORECASE) if p.strip()]
            else:
                phrases = [raw_term]

            reviews_df["review_text"] = reviews_df["review" if "review" in reviews_df.columns else "text"].fillna("")

            matched_sets = []
            for phrase in phrases:
                words = get_words(phrase)
                aspect_ids = set(aspects_df[
                    aspects_df.apply(lambda r: match_search_in_aspect_opinion(words, r["aspect"], r["opinion"]), axis=1)
                ]["id"].unique())
                review_ids = set(reviews_df[reviews_df["review_text"].str.contains(re.escape(phrase), case=False, na=False)]["id"].unique())
                matched_sets.append(aspect_ids | review_ids)

            if matched_sets:
                if logic_mode == "AND":
                    all_matched_ids = set.intersection(*matched_sets)
                elif logic_mode == "OR":
                    all_matched_ids = set.union(*matched_sets)
                else:
                    all_matched_ids = matched_sets[0]

            filtered_df = filtered_df[filtered_df["id"].isin(all_matched_ids)]

        st.markdown("---")
        st.success(f"✅ {len(filtered_df)} cars found matching your search criteria.")

        if len(filtered_df) == 0:
            st.warning("No cars match your current filters. Try broadening your search.")
            return

        # ---------------------------
        # Display matching cars
        # ---------------------------
        numeric_units = {
            "Horsepower": "hp",
            "Revolutions per minute (rpm)": "rpm",
            "Combined miles per gallon (mpg)": "mpg",
            "Cost": "USD",
            "0 to 60 time (seconds)": "s"
        }

        image_folder = "car_images"
        for _, row in filtered_df.iterrows():
            car_id = row["id"]
            make = row.get("Make", "") or "Unknown Make"
            model = row.get("Model", "") or "Unknown Model"
            model_name = f"{make} {model}".strip()
            st.markdown(f"### 🚘 {model_name} (ID: {car_id})")

            # Objective specs
            with st.expander("View Objective Specifications"):
                specs = {}
                for col in numeric_cols + categorical_cols:
                    if col in row:
                        val = row[col]
                        if pd.notna(val) and str(val).strip().lower() not in ["", "none", "nan", "not mentioned"]:
                            if col in numeric_units:
                                specs[col] = f"{val} {numeric_units[col]}"
                            else:
                                specs[col] = val
                if specs:
                    st.json(specs)
                else:
                    st.info("No valid specifications available for this car.")

            # ---------------------------
            # Corrected aspect/review highlighting with proper AND/OR
            # ---------------------------
            car_reviews = reviews_df[reviews_df["id"] == car_id]
            car_aspects = aspects_df[aspects_df["id"] == car_id]

            all_snippets = []
            displayed_any_snippet_for_car = False

            if search_term.strip() and phrases:
                search_words_per_phrase = [get_words(p) for p in phrases]

                for _, review_row in car_reviews.iterrows():
                    review_text = str(review_row.get("review", review_row.get("text", "")))

                    # --- Logical filtering first ---
                    phrase_match_flags = []
                    for words, phrase in zip(search_words_per_phrase, phrases):
                        aspect_opinion_matched = any(
                            match_search_in_aspect_opinion(words, arow['aspect'], arow['opinion'])
                            for _, arow in car_aspects.iterrows()
                        )
                        review_matched = re.search(re.escape(phrase), review_text, flags=re.IGNORECASE) is not None
                        phrase_match_flags.append(aspect_opinion_matched or review_matched)

                    if logic_mode == "AND" and not all(phrase_match_flags):
                        continue
                    if logic_mode == "OR" and not any(phrase_match_flags):
                        continue

                    # --- Highlight all matched aspects/opinions for all phrases ---
                    for _, arow in car_aspects.iterrows():
                        for words, phrase in zip(search_words_per_phrase, phrases):
                            if match_search_in_aspect_opinion(words, arow['aspect'], arow['opinion']):
                                highlighted_aspect = highlight_matching_words(arow['aspect'], words, color="#ffeb3b")
                                highlighted_opinion = highlight_matching_words(arow['opinion'], words, color="#90ee90")
                                st.markdown(
                                    f"**Aspect:** {highlighted_aspect}  |  **Opinion:** {highlighted_opinion}  | Sentiment: {arow.get('sentiment','')}",
                                    unsafe_allow_html=True
                                )
                                snippet = get_closest_aspect_opinion_snippet(review_text, arow['aspect'], arow['opinion'])
                                if snippet:
                                    all_snippets.append(snippet)
                                    displayed_any_snippet_for_car = True

                    # --- Highlight all phrases in the review text ---
                    
                   # --- Highlight only matched phrases in the review ---
                  
                    # Step 1: collect all snippets

                    for words, phrase in zip(search_words_per_phrase, phrases):
                        snippet_list = get_merged_highlighted_snippets(review_text, phrase, context=150)
                        all_snippets.extend(snippet_list)

            # Step 2: merge duplicates / highlights
            #all_snippets = merge_duplicate_snippets(all_snippets)
          
            # Final assemble of snippets (or fallback)
            if all_snippets:
                display_html = "<br><br> … <br><br>".join(all_snippets)
            else:
                #if not car_reviews.empty:
                 #   display_html = escape(str(car_reviews.iloc[0].get("review", car_reviews.iloc[0].get("text", ""))))
                #else:
                display_html = "<i>No reviews available.</i>"

            st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    max-height: 500px;
                    overflow-y: auto;
                    box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
                    font-size:15px;
                    color:#333;
                    line-height:1.5;
                ">
                    {display_html}
                </div>
            """, unsafe_allow_html=True)

            # Display image
            image_path = f"{image_folder}/{int(car_id) + 1}.jpg"
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.warning("📷 Image not found. (Run car_images_sample_Downloader.py)")

        # ---------------------------
        # Summary of matching cars
        # ---------------------------
        st.markdown("---")
        st.subheader("📋 Matching Cars Summary")
        unique_cars = filtered_df.drop_duplicates(subset=['id']).reset_index(drop=True)
        summary_lines = [f"{row['id']} — {row.get('Make','not mentioned')} {row.get('Model','not mentioned')}" for _, row in unique_cars.iterrows()]
        summary_text = "\n".join(summary_lines)
        st.text_area(f"{len(summary_lines)} Cars satisfying your query:", summary_text, height=150, disabled=True)

        
#if __name__ == "__main__":
#app()
