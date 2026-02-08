import streamlit as st

st.set_page_config(
    page_title="MuSe-CarASTE Launcher",
    layout="wide"
)
#st.set_option('deprecation.showPyplotGlobalUse', False)
#import warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
   


# ---------------------- Session State ----------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "prev_page" not in st.session_state:
    st.session_state["prev_page"] = st.session_state["page"]

# ---------------------- Sidebar Navigation ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Objective Features", "Subjective Features", "Compare Cars", "Car Search","Insight Portal"]  # 👈 added here
)

st.session_state["page"] = page

# ---------------------- Clear page-specific session keys on page switch ----------------------
if st.session_state["prev_page"] != page:
    st.session_state["prev_page"] = page
    
    # Remove old page-specific keys to avoid duplicate key errors
    keys_to_remove = [
    k for k in list(st.session_state.keys())
    if any(x in k for x in ["_obj", "_subj", "_comp", "_search","_ins"])  # 👈 added _search
    ]

    for k in keys_to_remove:
        del st.session_state[k]

# ---------------------- Pages ----------------------
if st.session_state["page"] == "Home":
    st.title("🎯 MuSe-CarASTE Launcher")
    st.write(
        """
        Welcome! Use the sidebar to navigate to a section:
        
        - **Objective Features:** Explore data-driven metrics.
        - **Subjective Features:** Explore opinion-based or sentiment analysis features.
        - **Compare Cars:** Perform detailed comparisons of cars on specifications and reviews.
        - **Car Search:** Quickly find cars based on your criteria.
        """
    )
    
    st.image("car1.jpeg", use_container_width=True)

elif st.session_state["page"] == "Objective Features":
    import Objective
    st.title("Objective Features")
    Objective.app()  # ✅ Only render Objective module

elif st.session_state["page"] == "Subjective Features":
    import Subjective
    st.title("Subjective Features")
    Subjective.app()  # ✅ Only render Subjective module

elif st.session_state["page"] == "Compare Cars":
    import CompareCars
    st.title("Compare Cars")
    CompareCars.app()  # ✅ New CompareCars module
elif st.session_state["page"] == "Car Search":
    import SearchCar
    st.title("Car Search")
    SearchCar.app()  # ✅ Render the CarSearch module
elif st.session_state["page"] == "Insight Portal":
    import Insight
    st.title("Get Ultimate Insight!")
    Insight.app()