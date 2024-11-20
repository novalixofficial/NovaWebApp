import streamlit as st
import pandas as pd
import tmap
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import os
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles, GetScaffoldForMol
from mapply.mapply import mapply
from rdkit.DataStructs import TanimotoSimilarity, ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
from venn import venn

import matplotlib.pyplot as plt
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
from rdkit.Chem import rdFingerprintGenerator
import joblib
from BackendAI import RemoveAutocorrelatedFeatures
from rdkit.DataStructs import BulkTanimotoSimilarity
#st.write("""## DEL Diversity & Addressability Analyzer​ """)
st.write("""## DEL Diversity & Addressability Analyzer
- Visualize the chemical space and  analyze the diversity of your libraries
- Evaluate the structural novelty of your scaffolds
- Predict the target addressability using machine learning """)
#st.write("Use this tool to visualize a TMAP diversity plot and a Venn diagram using your own libraries")
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# dictionary
nbits = 1024
fpdict = {}
#fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=1024)
#fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=1024)
fpdict['ecfp4'] = lambda m: rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits,includeChirality=True).GetFingerprint(m)
#fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=1024)
#fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
#fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
#fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
#fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
#fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=1024)
#fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=1024)
#fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=1024)
#fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
#fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
#fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
#fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=1024, nBitsPerHash=2)
#fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=1024, nBitsPerHash=2)
#fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=1024, nBitsPerHash=2)

default_color_list = ["#05BD1F","#0505BD","#F9A704","#F90426","#EC04F9","#F90426","#EFEF0B"]


def canon_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol:  # Check if the molecule is valid
            canon = Chem.MolToSmiles(mol)
            return canon
        else:
            print(f"{smi} is INVALID and will be removed!")
            return np.nan
    except Exception as e:
        print("Invalid smiles found. Please check that the SMILES column is correct.")
        return np.nan

def data_load():
### Asks for csv files to upload.
### Multiple files possible 
    #col1, col2 =st.columns([30,1],gap='small')
    placeholder = st.sidebar.empty()
    key_id = 0
    i = 0
    uploaded_files = []
    enum = [f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'}" for x in range(1, 21)]

 
    col_1, col_2, col_3, col_4,col_5 =st.columns([3.5,2.5,2.5,2,1],gap='medium')


    add_another = "No"
    valid_smiles = False
    #col = col1
    while i <= 4:
        with col_1:
            uploaded_file = st.file_uploader("FILE", type="csv", key=f"upload{key_id}",on_change=restart_session_state)
        key_id += 1
        i += 1
        if uploaded_file is not None:

            
            # Read the CSV file
            
            df = pd.read_csv(uploaded_file)
            list_col = list(df.columns)
            with col_2:
                st.write("SMILES column")
                smiles_col = st.selectbox("Select from the menu", list_col, key=f"smiles{key_id}",on_change=restart_session_state)
                st.write("####")
            iteration = enum[i]
            with placeholder.container():
                st.write(f""" #### Your {iteration} dataset uploaded is: """)
                st.write(df.head())
            
            
            if smiles_col:
                uploaded_files.append(uploaded_file)
                
                # Store the dataframe with the given SMILES column
                df["Structure"] = df[smiles_col]
                with col_3:
                    st.write("Library name")
                    df_label = st.text_input(f"Default is LIB-{i}", key=f"name{key_id}",value=f"LIB-{i}")
                    df_name = f"LIB-{i}"
                    st.write("####")

                if df_name:
                    df["Dataset"] = df_name
                                            
                        # Display the file name and dataframe
                    #st.write(f"**{df_name} ({uploaded_file.name})**")
                    
                    with col_4:
                        color = st.color_picker(f'Choose a color', key=f"color {df_name}",value=default_color_list[i-1])
                        
                        st.session_state.selected_colors[df_name] = color
                        st.session_state.lib_label[df_name] = df_label
                    
                        
                        if st.checkbox("""Check for invalid SMILES""", key=f"invalid_check {key_id}",value=False):
                            with st.spinner("Please wait ..."):
                                
                                if df_name not in st.session_state.canon:

                                    try:
                                        mol_test = Chem.MolFromSmiles(df.head(1)["Structure"].values[0])
                                        #st.success('SMILES read successfully', icon="✅")
                                        #st.write(mol_test)
                                    except Exception as e:
                                        st.error("Invalid SMILES. Please check SMILES column")
                                        break

                                    df["Molecule"] = df["Structure"].apply(lambda x: canon_smi(x))
                                    df = df.dropna(subset=["Molecule"])
                                    df.reset_index(drop=True, inplace=True)
                                    df["BM"] = df["Molecule"].apply(lambda x: MurckoScaffoldSmilesFromSmiles(x, includeChirality=True))
                                    st.session_state.canon[df_name] = df[["Molecule", "BM", "Dataset"]]
                            valid_smiles = True
                            st.write("#####")

                            if len(uploaded_files) > 0:
                                with col_5:
                                    if valid_smiles and len(uploaded_files) <= i:
                                        add_another = st.radio("Add more libraries", ("Yes", "No"), key=f"radio{key_id}",index=1)     # Ask if the user wants to upload another file
                                    st.write("######")
                                    st.write("###")
                                    if add_another == "No":
                                        placeholder.empty()
                                        break
                        else:
                            break
                
            else:
                st.warning("Please enter the SMILES column name.")
        else:
            break
    else:
        "Maximum of 5 libraries allowed to ensure visibility"



    #placeholder = st.sidebar.empty()
    #st.write(st.session_state.canon)

    return st.session_state.canon, st.session_state.selected_colors, st.session_state.lib_label

def murcko_remove_dup(df, struc_col, BM_col, dataset_name):
    df = df.drop(columns=[struc_col])
    df = df.drop_duplicates(subset=BM_col)
    df.reset_index(drop=True, inplace=True)
    #st.write(f"<h6>Number of unique BM Scaffolds for  {dataset_name} : {len(df)}<h6>", unsafe_allow_html=True)
    return df

def row_to_explicit_bit_vect(row):
    bit_vect = ExplicitBitVect(len(row))
    for i, bit in enumerate(row):
        if bit:
            bit_vect.SetBit(i)
    return bit_vect

def CalculateFP(smiles,fp_name='rdk5'): #handle INVALID smiles , replace with np.nan then drop later
    try:
        m = Chem.MolFromSmiles(smiles)
        fp = fpdict[fp_name](m)
        return np.array(fp)
    except Exception as e:
        print(f"WARNING: {e}")
        print(f'INVALID smiles detected: {smiles}')
        n_fp = len(fpdict[fp_name](Chem.MolFromSmiles('C')))
        return np.full((n_fp, ), np.nan)

def get_centroids_df(df,dataset_name,smiles_column="smiles",thresh=0.35,nbits=1024, fp='rdk5'):
    df = df.reset_index(drop=True)
    #st.write("Calculating Descriptors")

    fps_series = pd.DataFrame(df[smiles_column].apply(lambda x: CalculateFP(x, fp)).tolist(), columns=[f"bit_{i}" for i in range(nbits)])
    
    try:
        # Use mapply correctly with DataFrame's apply method
        fps = list(mapply(fps_series, lambda x: row_to_explicit_bit_vect(x), axis=1,progressbar=False))
    except Exception as e:
        st.write(f"Parallel processing failed with error: {e}")
        fps = fps_series.apply(lambda x: row_to_explicit_bit_vect(x), axis=1)
    
    
    #fps = list(mapply(fps_series, lambda x: row_to_explicit_bit_vect(x), axis=1,progressbar=False))
    
    #st.write("Clustering Started!")
    lp = rdSimDivPickers.LeaderPicker()

    picks = lp.LazyBitVectorPick(fps,len(fps),thresh)
    centroid =  list(picks)

    #st.write(f"<h6>Number of clusters for {dataset_name} (RDKit LeaderPicker): {len(centroid)}<h6>", unsafe_allow_html=True)
    nb_clg = len(centroid)
    df['centroid'] = df.index.to_series().apply(lambda x: x in centroid)
    #centroid_fps  = [fps[j] for j in picks]

    return df.query('centroid == True').reset_index(drop=True), nb_clg

def tmap_encoding(fingerprint, df, struc_col):
    bits = 1024
    mols = [Chem.MolFromSmiles(s) for s in df[struc_col]]
    fps = [fpdict[fingerprint](mol) for mol in mols]
    #user_lists = explicitVect_to_list(user_fps)
    lists = [tmap.VectorUchar(list(fp)) for fp in fps]
    enc = tmap.Minhash(bits)
    lf_fp = tmap.LSHForest(bits)
    lf_fp.batch_add(enc.batch_from_binary_array(lists))
    lf_fp.index()
    
    return lf_fp

def tmap_matplotlib(df, x, y, s, t, color_map, size, alpha, dataset_col, z_order={},lib_label={}):
    x = np.array(x)
    y = np.array(y)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10,10))
    
    Size= size *6
    #z_order = {"LIB1":2, "LIB2":1, "Preclinic":0}



   
    # Add scatter plot (points)
    for category, color in color_map.items():
        category_df = df[df[dataset_col] == category]
        ax.scatter(
            x[category_df.index],
            y[category_df.index],
            zorder= z_order[category] if len(z_order) != 0 else -len(category_df),
            s=Size,  # size of the markers
            c=color,  # color from color map
            label=lib_label[category] if len(lib_label) !=0 else None,
            alpha=float(alpha),  # transparency
            edgecolors='w',  # white edges for better visibility
            linewidth=0.5,
            marker='o'
        )
        
    # Add lines connecting points specified by s and t
    for from_point, to_point in zip(s, t):
        ax.plot(
            [x[from_point], x[to_point]],
            [y[from_point], y[to_point]],
            color='black',
            linewidth=0.2,  # line width
        )

    # Add legend
    ax.legend(title=dataset_col, fontsize=12)
    #ax.set_facecolor(bck_color)

    # Set title and labels
    #ax.set_title(plot_title, fontsize=10)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #fig.patch.set_facecolor('white')
    return fig

def Venn_diagram(df, chosen_colors, dataset_col, structure_col,lib_label):
    unique_keys = df[dataset_col].unique()
    if 1<len(unique_keys)<6:
        st.write("Venn Diagram shows the number of unique scaffolds and the common scaffolds among your libraries")
        df[structure_col] = df[structure_col].apply(lambda x: str(x) if not isinstance(x, str) else x)
        #data = {key: set(df[df[dataset_col] == key][structure_col]) for key in df[dataset_col].unique()}
        df[structure_col] = df[structure_col].astype(str)

        data = {lib_label[key]: set(df[df[dataset_col] == key][structure_col]) for key in unique_keys}
        #plt.figure(figsize=(15, 11))
        colors = [color for name, color in chosen_colors.items()]
        cmap = ListedColormap(colors)
        ax = venn(data, cmap=cmap)
        return ax.figure
    else:
        st.write(""" ###### You need at least two and maximum 5 datasets to generate a Venn diagram""")








def use_reference(BM_file, clustered_file, threshold, fingerprint, i):
    if fingerprint == "rdk5":
        ref_clg = pd.read_csv("preclinic_market_CUBM_rdk5.csv")
    else:
        ref_clg = pd.read_csv("preclinic_market_CUBM_ecfp4.csv")

    reference_BM = pd.read_csv("preclinic_market_UBM.csv")
    #reference = pd.read_csv("preclinic_market_UBM_delete.csv")

    BM_file = pd.concat((BM_file, reference_BM), ignore_index = True)
    BM_file.reset_index(drop=True, inplace=True)
    
    #ref_clg, nb_clg = get_centroids_df(reference, "Preclinic & Market",smiles_column="BM",thresh=threshold,nbits=1024, fp=fingerprint)
    if fingerprint == "rdk5":
        st.sidebar.write(f"<h6>Number of clusters for Preclinic & Market (RDKit LeaderPicker): 971<h6>", unsafe_allow_html=True)
    else:
        st.sidebar.write(f"<h6>Number of clusters for Preclinic & Market (RDKit LeaderPicker): 7776<h6>", unsafe_allow_html=True)
#    st.sidebar.write(f"<h6>Number of clusters for Preclinic & Market (RDKit LeaderPicker): {nb_clg}<h6>", unsafe_allow_html=True)
    clustered_file = pd.concat((clustered_file, ref_clg), ignore_index = True)
    clustered_file.reset_index(drop=True, inplace=True)
    
    full_reference = pd.read_csv("preclinic_market_Full.csv")

    return BM_file, clustered_file, full_reference


def get_cluster_ID(smile,smi_cent_list,centroid_fps,fp_name='rdk5'):
    #print("row",row)
    #smile = row[smiles_column]                
    exact_match_indices = np.argwhere(np.array(smi_cent_list) == smile).flatten()
    if len(exact_match_indices) > 0:
        return exact_match_indices[0]                    
    else:
        #fp = fps[row.name]                
        fp = row_to_explicit_bit_vect(fpdict[fp_name](Chem.MolFromSmiles(smile))) 
        max_index = np.argmax(np.array(BulkTanimotoSimilarity(fp, centroid_fps)))
        return max_index


def get_scaffold_addressability(df, prob_col="My_Property", cluster_ID="BM_scaffold", prob_thresh=0.5):
    # Sort the dataframe by the probability column in descending order
    df_sorted = df.sort_values(by=prob_col, ascending=False)
    
    # Calculate the cumulative sum of the probability column greater than the threshold
    n_positive = (df_sorted[prob_col] > prob_thresh).cumsum()
    print("Cumulative n_positive Done!")

    
    # Get the unique scaffold list and count of unique scaffolds cumulatively
    unique_scaffolds = (~df_sorted[cluster_ID].duplicated()).cumsum()
    print("Cumulative BM scaffold unique Done!")

    
    # Calculate the total number of rows
    n_total = len(df)
    
    scaff_addr = unique_scaffolds.values
    cmpd_addr = n_positive / n_total
    # Return the number of unique scaffolds and the normalized cumulative sum of positives
    return scaff_addr , cmpd_addr





def data_loader(i):
    col_1, col_2, col_3, col_4,col_5 =st.columns([3.5,3,3,2,0.5],gap='medium')
    with col_1:
        uploaded_file = st.file_uploader("FILE", type="csv", key=f"upload{i}")
        
        
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with col_2:
            st.write("SMILES column")
            smiles_column = st.selectbox("Select from the menu", list(df.columns), key=f"smiles{i}")
    
        with col_3:
            st.write("Library name")
            lib_name = st.text_input(f"Default is LIB-{i}", key=f"name{i}",value=f"LIB-{i}")  


        with col_4:
            color = st.color_picker(f'Choose a color', key=f"color {i}",value=default_color_list[i-1])
            add_more = st.checkbox("Add more library",key=f"add_{i}",value=False)

        #with col_5:
            


        return [df,smiles_column, lib_name, color,add_more]  


default_session_state = {
        "folder_name": "TMAP_output",
        "canon": {},
        "selected_colors": {},
        "df_BM": {},
        "df_clg": {},
        "csv_file": pd.DataFrame(),
        "BM_file": pd.DataFrame(),
        "struc_col": "Structure",
        "dataset_col": "Dataset",
        "clustered_file": pd.DataFrame(),
        "threshold" : float(),
        "fingerprint": str,
        "lf_ecfp4": None,
        "legend_labels": [],
        "size": 5,
        "z_order": {},
        "alpha" : 0.7,
        "datasets_values" : [],
        "reference": pd.DataFrame,
        "ref_clg": pd.DataFrame,
        "choice_ref": None,
        "marker": "Not done",
        "BM_preclinic": pd.DataFrame(),
        "clg_preclinic": pd.DataFrame(),
        "tmap_coords": [],
        "ready_step3":False,
        "X_desc_ML": [],
        "addr_results": {},
        "addr_df": pd.DataFrame(),
        "df_display": pd.DataFrame(),
        "lib_label": {},
        "full_ref": pd.DataFrame(),
    }

def restart_session_state():
    for key, value in default_session_state.items():
        st.session_state[key] = value


def main(): 

    np.random.seed(42)





    proceed_step2, proceed_step3, run_prediction = False, False, False
    
    # Initialize session state
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not os.path.exists(st.session_state.folder_name):
            os.makedirs(st.session_state.folder_name)
#            st.write(st.session_state.folder_name)
            
    st.write("""### Step 1: Upload your CSV file containing SMILES​ """)
    st.write("Please upload your .csv data files with the smiles of full compounds")
    st.write("One dataset per file. We suggest comparing up to two files for better interpretability")
                
    placeholder = st.sidebar.empty()
    st.session_state.canon, st.session_state.selected_colors, st.session_state.lib_label = data_load()
    
    #n_lib = len(st.session_state.canon)


    dataset_col = "Dataset"
    struc_col = "Molecule"
    BM_col = "BM"
    plt.style.use('ggplot')
    if st.session_state.canon:
        #st.write("⚠️Make sure the SMILES columns are correct before proceeding")
        proceed_step2 = st.checkbox("""Proceed to Step 2""", key = "files upload")
        if proceed_step2:
            st.write(" ----------------------------")
            col1, col2 = st.columns(2, vertical_alignment="top")
            with col1:
                st.write("""### Step 2: ​Cluster BM Scaffolds​ """)
                st.write("This step will generate BM Scaffold clusters satisfying a Tanimoto threshold")
                
                for dataset in st.session_state.canon.values():
                    name = dataset[dataset_col].unique()
                    name = name[0]
                    if name not in st.session_state.df_BM:
                        st.session_state.df_BM[name] = murcko_remove_dup(dataset, struc_col, BM_col, name)
                for df_BM in st.session_state.df_BM.values():
                    name = df_BM[dataset_col].unique()
                    name = name[0]
                    st.sidebar.write(f"<h6>Number of unique BM Scaffolds for  {name} : {len(df_BM)}<h6>", unsafe_allow_html=True)
                
                st.session_state.BM_file = pd.concat(st.session_state.df_BM.values(), ignore_index = True)
                st.session_state.BM_file.reset_index(drop=True, inplace=True)
                st.session_state.csv_file = pd.concat(st.session_state.canon.values(), ignore_index = True)
                st.session_state.csv_file.reset_index(drop=True, inplace=True)
                df = st.session_state.csv_file
                first_two = df.head(2)
                last_two = df.tail(2)
                df_overview = pd.concat([first_two, last_two])
                st.sidebar.write(df_overview)
                
                fp_list=[]
                fp_list = [str(x) for x in fpdict]
                st.session_state.fingerprint = st.selectbox('Select Fingerprint', fp_list)
               
                st.session_state.threshold = 0.35
                st.write("The default threshold is 0.65")
                #st.session_state.threshold = st.slider('Adjust threshold', 0.0, 1.0, 0.65)
                
                if st.checkbox("Start Clustering"):
                    with st.spinner("Please wait ..."):
                        for dataset in st.session_state.df_BM.values():
                            name = dataset[dataset_col].unique()
                            name = name[0]
                            if name not in st.session_state.df_clg:
                                st.session_state.df_clg[name], nb_clg = get_centroids_df(dataset,name,smiles_column=BM_col,thresh=st.session_state.threshold,nbits=1024, fp=st.session_state.fingerprint)
                                st.sidebar.write(f"<h6>Number of clusters for {name} (RDKit LeaderPicker): {nb_clg}<h6>", unsafe_allow_html=True)

                                
                        st.session_state.clustered_file = pd.concat(st.session_state.df_clg.values(), ignore_index = True)
                        st.session_state.clustered_file.reset_index(drop=True, inplace=True)
                        #st.write("Complete clustered file :", len(st.session_state.clustered_file))
                        
    ### Use now st.session_state.clustered_file
    ### It contains the concatenation of all clusters of BM Scaffolds of each dataset_col
    ### For Venn Diagram: use st.session_state.BM_file (to check exact matches in BM Scaffolds)
    ### st.session_state.BM_file contains concat of all BM uniques of each dataset
                        
                    
                    
                    
                    #if st.checkbox("Compare my data with the Preclinic & Market"):
                    #st.session_state.choice_ref = st.radio("Choose an option: ",("Use only my data", "Compare my data with the Preclinic & Market"))
                    st.session_state.choice_ref = st.toggle("Compare my data with the Preclinic & Market",on_change=restart_session_state,value=True)
                    
                    i=0
                    if st.session_state.choice_ref:
                        if st.session_state.marker == "Not done" :
                            with st.spinner("Please wait ..."):
                        #st.session_state.choice_ref == "Compare my data with the Preclinic & Market":
                                st.session_state.BM_preclinic, st.session_state.clg_preclinic, st.session_state.full_ref = use_reference(st.session_state.BM_file, st.session_state.clustered_file, st.session_state.threshold, st.session_state.fingerprint, i=0)
                                st.session_state.lib_label["Preclinic & Market"]="Preclinic & Market"
                                st.session_state.marker = "Preclinic & Market clustering already done"
                        st.session_state.selected_colors["Preclinic & Market"] = st.color_picker('Choose a color for Preclinic & Market', key="color ref",value="#D3D4CE")
                        #st.session_state.selected_colors["Preclinic & Market"] = ref_color
                        st.session_state.clustered_file = st.session_state.clg_preclinic
                        st.session_state.BM_file = st.session_state.BM_preclinic
            
                    if st.session_state.full_ref is not None:
                        st.session_state.csv_file = pd.concat([st.session_state.csv_file, st.session_state.full_ref], ignore_index = True)    
                    st.write("   ")
                    proceed_step3 = st.checkbox("Proceed to Step 3", key = "tmap_venn")
                    if proceed_step3:
                        #st.write(" ----------------------------")
                        with col1:
                            


                            if st.session_state.lf_ecfp4 == None:
                                lf_fp = tmap_encoding(st.session_state.fingerprint, st.session_state.clustered_file, BM_col)
                                st.session_state.lf_ecfp4 = lf_fp

                            if st.session_state.lf_ecfp4:
                                if len(st.session_state.tmap_coords) == 0:
                                    with st.spinner("Calculating TMAP coordinates"):
                                        x, y, s, t, _ = tmap.layout_from_lsh_forest(st.session_state.lf_ecfp4)
                                        st.session_state.tmap_coords = [x,y,s,t]


                                label_encoder = LabelEncoder()
                                st.session_state.clustered_file["label"] = label_encoder.fit_transform(st.session_state.clustered_file[dataset_col])
                                legend_labels = list(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
                                st.session_state.legend_labels = legend_labels
                                if len(st.session_state.tmap_coords) > 0:
                                    st.session_state.ready_step3 = True
                                    
                        st.write("   ")
                        #st.write(" ----------------------------")


    if st.session_state.ready_step3 and proceed_step2 and proceed_step3:
        st.write(" ----------------------------")
        st.write("""### Step 3: Plot TMAP & Venn Diagram​ """)
        col_3_1, col_3_2 = st.columns(2, vertical_alignment="top",gap='medium')
                                
        
        with col_3_1:
            st.write("TMAP visualizes the chemical space as trees. Large coverage suggests high diversity.")
            x,y,s,t = st.session_state.tmap_coords
            #st.write(st.session_state.lib_label)
            fig = tmap_matplotlib(st.session_state.clustered_file, x, y, s, t, st.session_state.selected_colors, st.session_state.size, st.session_state.alpha, dataset_col, st.session_state.z_order,st.session_state.lib_label)
            st.pyplot(fig)
                                    
        with col_3_2:
            figure = Venn_diagram(st.session_state.BM_file, st.session_state.selected_colors, dataset_col, BM_col,st.session_state.lib_label)
            if isinstance(figure, plt.Figure):
                st.pyplot(figure)
        

        if st.checkbox("Customize Plots (Optional)", key = "customize_plot"):

            col_3_3, col_3_4 = st.columns(2, vertical_alignment="top",gap='medium')

            with col_3_3:

                st.session_state.size = st.slider("Adjust point size", 1, 20, 5, key="Slider point size")
                st.session_state.alpha = st.slider("Adjust transparency: 0 is transparent, 1 is opaque", 0.0, 1.0, 0.7)
                                    
                st.session_state.datasets_values = st.session_state.clustered_file[dataset_col].unique()
                st.write("   ")
            
            st.button("Update Plot", key="update_plot", help=None, on_click=None)

                                        
        st.write("    ")


        if st.checkbox("Proceed to Step 4", key="Step 4 Addressability"):
            st.write("----------------------------")
            st.write("""### Step 4: Predict addressability """)
            st.write("Upload here your own machine learning model to predict the addressability of your compounds")
            ML_file = st.file_uploader("ML Model (**.joblib file trained using NovaML**)", type=["joblib","pkl"], key="ML Model upload",on_change=restart_X_desc_ML)
            cluster_col = "cluster_ID"
            if ML_file:
                model =  joblib.load(ML_file)

                #st.write(ML_model)

                
                #st.dataframe(st.session_state.canon['LIB-1'])

                smiles_column = "Molecule"
                
                #df_desc = np.array(list(mapply(df_i[smiles_column], lambda x: CalculateFP(x,'ecfp4'), progressbar=False))).astype(np.uint8)

                #df_i["is_BRD4"] = ML_model.predict_proba(df_desc)[:,1]
                #st.dataframe(df_i)

                addr_df =  st.session_state.csv_file.query(f"{dataset_col} != 'Preclinic & Market'").copy()         
                #st.write(addr_df)  
                    

                from sklearn.base import is_classifier, is_regressor
                from BackendAI import get_desc            

                task_type = "Classifier" if is_classifier(model.named_steps["estimator"]) else "Regressor"

                desc_list = ["ECFP4", "FCFP4", "RDK5","RDKIT"]

                desc_choice = [desc for desc in desc_list if desc in ML_file.name.split('_')][0]

                if desc_choice:

                    st.write(f"The model is a **{task_type}** using the **{desc_choice}** descriptor")
                else:
                    st.write("Could not automatically identify the molecular features used. Please specify")
                    desc_choice = st.selectbox("Select descriptor used", tuple(["ECFP4", "FCFP4", "RDK5","RDKIT"]))  

                var_pred = st.text_input("Give the name of the variable to be predicted", value="My_Property")
                run_prediction = st.checkbox("Run Predictions", key="run_prediction",on_change=restart_addr_results)
            
            if run_prediction and ML_file:
                if len(st.session_state.X_desc_ML) == 0:
                    with st.spinner('Calculating Descriptors...'):
                        st.session_state.X_desc_ML = get_desc(smi_series=addr_df[smiles_column],desc_name=desc_choice)

                
                if len(st.session_state.addr_results) == 0:
                    with st.spinner('Performing Inference...'):
                        col_4_1, col_4_2 = st.columns(2, vertical_alignment="top",gap='medium')
                        try:
                            if task_type == "Classifier":
                                addr_df[var_pred] = model.predict_proba(st.session_state.X_desc_ML)[:,1]
                                addr_df[var_pred] = addr_df[var_pred].apply(lambda x: np.round(x,decimals=4))

                            else:
                                addr_df[var_pred] = model.predict(st.session_state.X_desc_ML)
                                addr_df[var_pred] = addr_df[var_pred].apply(lambda x: np.round(x,decimals=4))
                                                     
                        except Exception as e:
                            st.write(e)
                            st.write("Error. Please check for INVALID SMILES or if the descriptor used was the correct one")


                    
                            
                if var_pred in addr_df.columns:
                    
                    with col_4_1:
                        if task_type == "Classifier":
                            addr_threshold = st.slider('Probability threshold for the positive class (default is 50%)', 0.1,0.9, 0.5,
                                                       on_change=restart_addr_results)  

                        else:
                            addr_threshold = st.slider('Value threshold for the positive class (default is the median)', addr_df[var_pred].min(),
                                                        addr_df[var_pred].max(), addr_df[var_pred].median(),
                                                        on_change=restart_addr_results)

                    
                plot_addr = st.checkbox("Generate Addressibility Plot", key="addr_plot")  
                if plot_addr: 
                    if len(st.session_state.addr_results) == 0 and plot_addr:
                        with st.spinner('Generating Plot...'):  
                            addr_df[cluster_col] = np.nan

                            
                            for dataset in addr_df[dataset_col].unique():
                                
                                df_dataset = addr_df.loc[addr_df[dataset_col]== dataset]
                                fp_name = st.session_state.fingerprint
                                #fp_name = 'rdk5'

                                df_centroid =  st.session_state.clustered_file.loc[st.session_state.clustered_file[dataset_col]== dataset]

                                smi_centroid_list = list(df_centroid[BM_col])

        
                                fps_series =  pd.DataFrame(list(df_centroid[BM_col].apply( lambda x: CalculateFP(x,fp_name))),columns=[f"bit_{i}" for i in range(nbits)])                  
                                centroid_fps = list(fps_series.apply(lambda x: row_to_explicit_bit_vect(x), axis=1 ))
                                cluster_ID = list(df_dataset["BM"].apply(lambda smi: get_cluster_ID(smi,smi_centroid_list,centroid_fps,fp_name) ))  
                                addr_df.loc[addr_df[dataset_col]== dataset,cluster_col] = cluster_ID
                                
                                df_dataset = addr_df.loc[addr_df[dataset_col]== dataset]
                                scaff_addr, cmpd_addr = get_scaffold_addressability(df_dataset, prob_col=var_pred, cluster_ID=cluster_col, prob_thresh=addr_threshold)
                                st.session_state.addr_results[dataset] = [scaff_addr,cmpd_addr]
                                
                            
                            if len(st.session_state.addr_df) == 0:
                                
                                st.session_state.addr_df = addr_df.round({var_pred: 4, cluster_col: 0})
                                #st.dataframe(st.session_state.addr_df)
                            

                            
                        
                            
                if len(st.session_state.addr_results) > 0 and plot_addr: #model predictions is already done
                    #st.write(st.session_state.csv_file)
                    #if len(st.session_state.addr_results) == 0:
                    #    for dataset in st.session_state.csv_file[dataset_col].unique():
                            
                    col_4_3, col_4_4 = st.columns(2, vertical_alignment="top",gap='medium')    
                    with col_4_3:            
                        st.pyplot(addressability_plot(st.session_state.addr_results,st.session_state.selected_colors,st.session_state.lib_label))
                    with col_4_4:
                        st.write("#####")
                        st.write("""**Scaffold-based addressability**: number of distinct scaffolds oriented to your specific target""")
                        st.write("""**Compound-based addressability**: fraction of full compounds oriented to your specific target""")
                        st.write("""For more information, please refer to our article.""")
                    #st.dataframe(st.session_state.csv_file)
                    st.write("#####")
                    st.write("#### Examine the addressability of your compounds")

                    cmpd_to_display = st.radio("", ("All compounds", "Optimized diversity (only the best-scoring compounds for each scaffold cluster)"), key=f"radio_all_or_cluster",index=0,on_change=restart_df_display)  

                    if len(st.session_state.df_display) == 0:
                        if cmpd_to_display == "All compounds":
                            df_display =  st.session_state.addr_df.sort_values(by=var_pred,ascending=False)
                        else:
                            df_display = st.session_state.addr_df.sort_values(var_pred, ascending=False).groupby(["cluster_ID","Dataset"]).agg({i:"first" for i in st.session_state.addr_df.columns }).sort_values(var_pred, ascending=False).reset_index(drop=True)
                        st.session_state.df_display = df_display

                    tab_1_1, tab_1_2=  st.tabs(["Molecule View", "Table"])

                    import mols2grid
                    import streamlit.components.v1 as components

                    
                    with tab_1_1:
                        #st.session_state.input_df["SMILES"] =  st.session_state.input_df[smi_col]

                        raw_html = mols2grid.display(st.session_state.df_display,
                                                    subset=[var_pred],
                                                    smiles_col=smiles_column,
                                                    tooltip=[i for i in st.session_state.df_display.columns.to_list() if i not in [smiles_column,"smiles","SMILES","canon_smi"] ],
                                                    size=(208,104))._repr_html_()
                        components.html(raw_html, height=104*7, scrolling=True)              

                    with tab_1_2:

                        st.dataframe(st.session_state.df_display,height=104*7) 


    
                    if len(st.session_state.addr_df) > 0:
                        col_4_5, col_4_6, col_4_7 = st.columns([1,1,6], vertical_alignment="top",gap='medium')

                        with col_4_5: 
                            st.download_button(
                            "Download as **CSV**",convert_df(st.session_state.df_display),"file.csv","text/csv",key='download-addr-csv')  
                        with col_4_6:

                            @st.cache_data
                            def download_sdf(df_display,smiles_column):
                                df_display["mol"] = df_display[smiles_column].apply(Chem.MolFromSmiles)
                                return Chem.PandasTools.WriteSDF(df_display,"./file_download.sdf", molColName='mol', idName='cluster_ID', properties=df_display.columns, allNumeric=False, forceV3000=False)
                            
                            download_sdf(st.session_state.df_display,smiles_column)

                            with open("./file_download.sdf", "rb") as fp:
                                btn = st.download_button(
                                    label="Download as **SDF**",
                                    data=fp,
                                    file_name="file_download.sdf",
                                    mime="application/sdf")
                            
                            #st.button("Download SDF", key="download-addr-sdf", help=None, on_click=download_sdf)
                            

     
            
        #mod_predicter.proceed_step2 = st.checkbox("I confirm that the model file and the target variable name is correct.")                        


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')




def addressability_plot(addr_results,chosen_colors,lib_label):
    plt.style.use('default')
    #import seaborn as sns
    #sns.set_theme()
    #sns.set_style("ticks")
    #sns.despine()
    #n_total = len(freq_scaff_1)
    colors = [color for name, color in chosen_colors.items()]

    fig, ax = plt.subplots(figsize=(5,4))
    for addr_name,color in zip(addr_results.keys(),colors):
        scaff_addr, cmpd_addr = addr_results[addr_name]
        ax.plot(scaff_addr,cmpd_addr,color=color)
        ax.fill_between(scaff_addr,cmpd_addr,label=lib_label[addr_name],color=color,alpha=0.25)
       
        #plt.plot(x_ideal/n_total,x_ideal/n_total,ls=':',color='gray')
        #plt.xlim([0,100])
        #plt.ylim([0,100])
    plt.legend(fontsize=12,loc='best')
    plt.ylabel("Compound-based\n Addressability ",fontsize=15)
    plt.xlabel("Scaffold-based Addressability",fontsize=15)

        #plt.xticks(np.arange(0,7000, 1000),fontsize=15)

    return fig
                
def restart_X_desc_ML():
    st.session_state.X_desc_ML = []

def restart_addr_results():
    st.session_state.addr_results = {}

def restart_df_display():
    st.session_state.df_display = pd.DataFrame()
    



                
                
                
                
                
                
                
                
                
                
                
                
                
                
#st.write("""### Step 4: ​Predict Addressability​ """)


        
main()

