import streamlit as st


def home_page():
    st.markdown("######")
    st.image("Novalix_logo_green.png", use_column_width=False)
    st.title("Welcome to our Cheminformatic Web App")
    
    
    #st.write("This app has the following modules")
    st.markdown("#### NovaML")
    #st.write("Build AI/ML models using your own molecular data or predict the target addressability of your libraries")
    st.write("- Build AI/ML models using your own molecular data")
    st.write("- Predict the target addressability of your libraries")
    st.markdown("#### NovaDEL Analyzer")
    #st.write("Visualize the chemical space and analyze both the diversity and target addressability of your libraries")
    st.write("- Visualize the chemical space and  analyze the diversity of your libraries")
    st.write("- Evaluate the structural novelty of your scaffolds")

    st.markdown('---')
    st.info("""If you used our tools in your publications, please consider citing this article:  
            Evaluating the Diversity and Target Addressability of DELs using BM-Scaffold Analysis and Machine Learning.
                 **Manuscript Submitted**""", icon="ℹ️")
    

def page2():
    st.title("NovaDEL Analyzer")

def page4():
    st.title("About Novalix")
    novalix_url = "https://novalix.com/who-we-are/"
    linkedin_url = "https://fr.linkedin.com/company/novalix"
    #st.write("check out this [link](%s)" % novalix_url)
    st.markdown("##### [Website](%s)" % novalix_url)    
    st.markdown("##### [Linkedin](%s)" % linkedin_url)    




st.logo("Novalix_logo_green.png")


st.set_page_config(layout='wide',page_icon="novalix_icon.png")

# Define all the available pages, and return the current page
current_page = st.navigation({
    "Welcome": [st.Page(home_page, title="Home", icon="🏠")],

    "Modules": [
        st.Page("NovaML.py", title="NovaML", icon="💻"),
        st.Page("DEL_analyzer.py", title="NovaDEL Analyzer", icon="⚛️"),
    ],
    "About Novalix": [
        st.Page(page4, title="Novalix", icon=":material/thumb_up:")
    ],
})

# current_page is also a Page object you can .run()
current_page.run()
