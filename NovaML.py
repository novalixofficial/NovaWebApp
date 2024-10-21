######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
import mols2grid
import streamlit.components.v1 as components

import plotly.express as px
import datetime
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr, friedmanchisquare, wilcoxon
import scikit_posthocs as sp

import sys,os
import base64
from io import BytesIO
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.getcwd()
# Insert functions path into working dir if they are not in the same working dir
sys.path.insert(1, cwd)
from BackendAI import benchmark_model, get_desc,eval_predictions
#from custom_classes import RemoveZeroVarianceFeatures, RemoveAutocorrelatedFeatures
#from rdkit.Chem import Draw
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"
font_style = dict(family="Verdana",size=23,color="black")




if 'input_df' not in st.session_state:
    st.session_state.input_df = pd.DataFrame()

if 'train_df' not in st.session_state:
    st.session_state.train_df = pd.DataFrame()

if 'benchmark_obj' not in st.session_state:
    st.session_state.benchmark_obj = object()

if 'desc_dict' not in st.session_state:
    st.session_state.desc_dict = {}

if 'data_obj' not in st.session_state:
    st.session_state.data_obj = {}

if 'cv_df' not in st.session_state:
    st.session_state.cv_df = pd.DataFrame()

if 'X_desc_test' not in st.session_state:
    st.session_state.X_desc_test = np.array([])

if 'test_file' not in st.session_state:
    st.session_state.test_file = pd.DataFrame()


def restart_session_state():
    st.session_state.hyperopt_df = pd.DataFrame()
    st.session_state.train_df = pd.DataFrame()
    st.session_state.desc_dict = {}
    st.session_state.X_desc_test = np.array([])
    st.session_state.test_file = pd.DataFrame()



#logo = Image.open('novalix_logo_small.png')
#st.image(logo, use_column_width=False)
st.write("""
## Nova-ML: Machine Learning for Molecular Data
Build ML models or perform inference using your own csv data 
""")


st.write("""#### What do you want to do?""")

class mod_builder:

    #initialize bools
    objective = None
    column_name_check = False
    valid_csv = False
    filter_csv=False
    plot_scatter = False
    get_benchmark = False
    get_unwanted_smarts = False
    get_physchem_limits = False
    get_output_file = False
    visualize = False
    #plot_benchmark = False
    improve_model = False
    proceed_step2 = False
    proceed_step3 = False
    proceed_step4 = False
    proceed_step4 = False
    proceed_step5 = False
    run_bayesian_opt = False
    plot_histogram = False
    task = None
    split_type = None
    binary_threshold = None
    show_advanced_viz = False






mod_builder.objective = st.radio(
    "",
    ["Train ML models", "Predict properties"],
    captions = ["Train your own ML model (Random Forest, LightGBM, XGBoost)", "Get predictions for your compounds using a pre-built model"],on_change=restart_session_state)


st.markdown("""**Note:** This is a simplified user-friendly machine learning workflow with no coding/programming required.
            To professionally build your ML models or explore more advanced deep learning models (e.g.  chemprop D-MPNN, CNN, Transformers), contact our cheminformatics team.""")



def plot_scatter_regr(df,x='observed',y='predicted',color="set",color_continuous_scale=None):              
    fig = px.scatter(df, x=x, y=y,color=color,color_continuous_scale=color_continuous_scale)
                    #fig_scatter.add_traces(list(px.line(df_scatter,x='observed',y='observed').select_traces()))
    parity_line = go.Scatter(
        x=[df[x].min()-np.abs(0.1*df[x].min()), df[x].max()+np.abs(0.15*df[x].max())],
        y=[df[y].min()-np.abs(0.1*df[y].min()), df[y].max()+np.abs(0.15*df[y].max())],
        mode='lines',
        name='Parity Line',
        line=dict(color='green', width=2))
    fig.add_trace(parity_line)        
    fig.update_layout(xaxis=dict(scaleanchor="x", titlefont=dict(size=18,color='black'), scaleratio=1,constrain="domain"),  # Lock aspect ratio
        yaxis=dict(scaleanchor="x",titlefont=dict(size=18,color='black'), scaleratio=1,constrain="domain"),   # Lock aspect ratio
        height=500)
    return fig


def plot_scatter_clf(df,x='observed',y='predicted',color="set"):
    df[x] = df[x].astype(bool)
    fig = px.strip(df, x=x, y=y,color=color)   
    fig.update_traces(opacity=.8)  
    fig.update_xaxes(categoryorder='array', categoryarray= [False,True])
    fig.update_layout(
        title="",
        xaxis_title="Experimental Classification (positive class)",
        yaxis_title="Probability for Positive Class",
        legend_title="Set",
        font=dict(family="Arial",size=18,color="RebeccaPurple"))
    fig.add_hrect(y0=0.490,y1=0.501)
    fig.add_vrect(x0=0.490,x1=0.501)
    annot_good = dict(color="#388E3C")
    annot_bad = dict(color="#B71C1C")
    fig.add_annotation(x=0, y=-0.1,text="True Negative",showarrow=False,arrowhead=1,font=annot_good)
    fig.add_annotation(x=1, y=-0.1,text="False Negative",showarrow=False,arrowhead=1,font=annot_bad)
    fig.add_annotation(x=1, y=1.1,text="True Positive",showarrow=False,arrowhead=1,font=annot_good)
    fig.add_annotation(x=0, y=1.1,text="False Positive",showarrow=False,arrowhead=1,font=annot_bad)

    return fig








if mod_builder.objective  == "Train ML models":

    st.markdown('---')
    st.write("""#### Step 1: Upload your CSV file containing SMILES and the target variable """)

    csv_file = st.file_uploader('Select File',type='.csv',on_change=restart_session_state)
    df = pd.DataFrame()

    if csv_file:
        if len(df) <= 10000:
            df = pd.read_csv(csv_file,engine='python')
        else:
            df = pd.read_csv(csv_file).sample(n=10000).reset_index(drop=True)  
        
        col_1_1, col_1_2 = st.columns([3,3],gap='medium')
        
        with col_1_1:    
            
            smi_col = st.selectbox("""**Select SMILES column**  
                                   Text representation of your molecules""",tuple(df.columns))
            
        with col_1_2:
            y_col = st.selectbox("""**Select Y column**  
                                 The target variable to predict, must be a number""", tuple([i for i in df.columns if i not in smi_col]))  

        mod_builder.column_name_check = st.checkbox("I confirm, the column names are correct.",on_change=restart_session_state)  

        if mod_builder.column_name_check :
            st.session_state.input_df = df

            #Check SMILES validity

            if st.session_state.input_df[smi_col].dtype == 'object':
                #st.write("Checking for invalid SMILES")                
                canon_smi = []
                for i,smi in enumerate(st.session_state.input_df[smi_col]):
                    try:
                        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi,sanitize=True))
                        canon_smi.append(canon) 
                    except:
                        canon_smi.append(np.nan)
                        st.error(f"Invalid SMILES {smi} with index {i} invalid and was removed")
                #
                st.session_state.input_df[smi_col] = canon_smi
                st.session_state.input_df = st.session_state.input_df.dropna(subset=smi_col).reset_index(drop=True) 
        
            else:
                st.error("**❗Error:** Please check that the smiles column contain only SMILES")

            #Check target column validity

            if is_numeric_dtype(st.session_state.input_df[y_col]):
                df[y_col] = df[y_col].astype(float)
                st.session_state.input_df[y_col] = df[y_col]
            else:
                st.markdown('######')
                st.error("**❗Error:** Target variable **must be a number**, if it is categorical, modify the CSV file and convert it to 0 and 1") 
                                        
             

            if st.session_state.input_df[smi_col].dtype == 'object' and is_numeric_dtype(st.session_state.input_df[y_col]):   
                
                st.success("✅ Input CSV file read successfully!")            

                tab_1_1, tab_1_2=  st.tabs(["Molecule View", "Table"])

                with tab_1_1:
                    #st.session_state.input_df["SMILES"] =  st.session_state.input_df[smi_col]

                    raw_html = mols2grid.display(st.session_state.input_df,
                                                 subset=[y_col],
                                                 smiles_col=smi_col,
                                                 tooltip=[i for i in st.session_state.input_df.columns.to_list() if i not in [smi_col,"smiles","SMILES","canon_smi"] ],
                                                 size=(208,104))._repr_html_()
                    components.html(raw_html, height=104*7, scrolling=True)              

                with tab_1_2:
                    st.dataframe(st.session_state.input_df,height=104*7)


    if mod_builder.column_name_check and is_numeric_dtype(st.session_state.input_df[y_col]) :
        col_1_4, col_1_5 = st.columns(2,gap='medium') 
        with col_1_4:
            task = st.radio("**Select a Task**",
                ["Regression", "Binary Classification"],
                captions = ["**Predict numerical values (when your data is quantitative / low noise)**", "**Predict classes/categories (when your data is qualitative / high noise)**"],on_change=restart_session_state)
            if task == "Binary Classification":
                
                mod_builder.binary_threshold = st.slider('Binary Classification threshold (used to define class labels "low" or "high")', df[y_col].min(), df[y_col].max(), df[y_col].median())


            mod_builder.plot_histogram = st.checkbox("Validate threshold and plot histogram")    
            if mod_builder.plot_histogram:
                #nbins = st.slider('Number of Histogram Bins', 2, 50, 20)

                if task == "Binary Classification":
                
                    df = st.session_state.input_df
                    df['cat'] = df[y_col] > mod_builder.binary_threshold
                    df['classification'] = df['cat'].apply(lambda x: "high" if x == True else "low" )                
                
                    frac_positive = len(df[df.cat == 1])/len(df)*100
                    st.write(f"Number of positive class: {frac_positive:.1f}% of {len(df)}")
                    fig = px.histogram(df, x=y_col, hover_data=df.columns,nbins=20,color='classification')
                    
                else:
                    fig = px.histogram(df, x=y_col, hover_data=df.columns,nbins=20)
                st.plotly_chart(fig)


        if mod_builder.plot_histogram:
            mod_builder.proceed_step2 = st.checkbox("Proceed to Step 2")
    else:
        print("Waiting for CSV file")

if mod_builder.proceed_step2 == True:
    
    st.markdown('---')
    st.write("""#### Step 2: Select Descriptors, Algorithms, and Data Splits """)


    col_2_1, col_2_2 = st.columns(2,gap='medium')

    with col_2_1:    
        desc_choice = st.multiselect(
                        "Choose your Descriptors",
                        ["ECFP4", "FCFP4","RDKIT"],
                        ["ECFP4","FCFP4","RDKIT"],on_change=restart_session_state)

    with col_2_2:
        algo_choice = st.multiselect(
                        "Choose your Algorithms",
                        ["RandomForest", "LightGBM", "XGBoost","ExtraTrees","Consensus"],
                        ["LightGBM","XGBoost"],on_change=restart_session_state)
        
    st.write("Parameterize your data splits")
    col_2_3, col_2_4 = st.columns(2,gap='medium')
    with col_2_3:
        test_size = st.slider('Test Size (Default is 20%)', 0, 100, 20)
    
    with col_2_4:
        CV_split = st.slider('Number of Cross-validation Cycles (Default is 10)',5,30,10,step=5)
    
    
    
    mod_builder.split_type = st.radio(
    "",
    ["Random Split", "Scaffold Split"],
    captions = ["Test/Validation sets will be randomly selected", "Test/Validation sets will not contain the same scaffolds as the training set"],on_change=restart_session_state)

    mod_builder.proceed_step3 = st.checkbox("Proceed to Step 3")


if mod_builder.proceed_step3 == True:

    st.markdown('---')
    st.write("""#### Step 3: Evaluate Model Performance """)



    smi_series = st.session_state.input_df[smi_col]
    if task == "Binary Classification":
        #if 'cat' in st.session_state.input_df:
        target_y = st.session_state.input_df['cat']
    else:
        target_y = st.session_state.input_df[y_col]

    #@st.cache_data


    # Train models (this step will run every time changes are made)
    if len(st.session_state.train_df) == 0:

        with st.spinner('Training your machine learning models...'):
            st.session_state.train_df, st.session_state.data_obj, st.session_state.cv_df = benchmark_model(smi_series=smi_series, y=target_y, desc_list=desc_choice, 
                                                        model_name=algo_choice,task=task,test_size=test_size,
                                                        CV_split=CV_split, split_type=mod_builder.split_type)
    


    def plot_barplot(df,x='desc',y='test_roc_auc',xlabel="descriptor",ylabel="AUC",xlim=None,ylim=None,title='Title'):
        color = "desc" if x=="algo" else "algo"
        if title=="Cross-validation Scores":
            fig = px.bar(df,x=x,y=y,error_y=f"{y}_std",color=color,barmode='group',height=500,title=title)  
        else:
            fig = px.bar(df,x=x,y=y,color=color,barmode='group',height=500,title=title)       
        fig.update_yaxes(range=ylim,linewidth=3,tickwidth=3)
        fig.update_xaxes(range=xlim,linewidth=3,tickwidth=3)
        
        fig.update_layout(
            title="",xaxis_title=xlabel,yaxis_title=ylabel,
            font=font_style)
        fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.4,
            'xanchor': 'center',
            'yanchor': 'top'})

        return fig


    def plot_boxplot(df,x='desc',y='test_roc_auc',xlabel="descriptor",ylabel="AUC",xlim=None,ylim=None):
        color = "desc" if x=="algo" else "algo"
        fig = px.box(df, x=x, y=y, color=color,
             notched=False, # used notched shape
             hover_data=[y] # add day column to hover data
            )

        def get_p_value(df,color,x_i):
            n_subgroup = len(df[color].unique()) 
                
            if n_subgroup == 2:
                subgroups = [df[df[x] == x_i][df[color] == color_i][y].values  for color_i in df[color].unique()]
                p_value = wilcoxon(subgroups[0],subgroups[1]).pvalue
            elif n_subgroup >= 3:
                subgroups = [df[df[x] == x_i][df[color] == color_i][y].values  for color_i in df[color].unique()]
                p_value = friedmanchisquare(*subgroups).pvalue
            else:
                p_value = np.nan
            return p_value
            
        annotations = [
            dict(
                x=x_i, 
                y=1.1*df[df[x] == x_i][y].max(), 
                text=f"p={get_p_value(df,color,x_i):.3f}", 
                showarrow=False,
                arrowhead=0
            )
            for i,x_i in enumerate(df[x].unique())
        ]

        fig.update_layout(annotations=annotations,font=dict(size=13,color="RebeccaPurple"))
        fig.update_yaxes(range=ylim,linewidth=3,tickwidth=3,title_font=dict(size=12))
        fig.update_xaxes(range=xlim,linewidth=3,tickwidth=3,title_font=dict(size=15),tickfont = dict(size=15,color='black'))

        return fig


    def get_signplot(df,x='desc',y='test_AUC'):
        num_subplots = len(np.unique(df[x]))
        fig, axes = plt.subplots(1, num_subplots, figsize=(9, 3),layout="constrained")

        # Populate each subplot
        for i,x_i in enumerate(df[x].unique()):
            ax = axes[i]

        
            group_col = "desc" if x=="algo" else "algo"
            df_i = df[df[x] == x_i ]
            n_cv = df_i[df_i[group_col] == np.unique(df_i[group_col])[0]].shape[0]
            n_groups = len(np.unique(df_i[group_col]))
            df_i["cv_cycle"] = df_i.groupby([group_col]).cumcount() + 1
            sns.set(rc={'figure.figsize': (3, 3)}, font_scale=1)
            heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': True, 'square': True}
            pc = sp.posthoc_conover_friedman(df_i, y_col=y, group_col=group_col, block_col="cv_cycle", p_adjust="holm",melted=True)

            sub_ax, sub_c = sp.sign_plot(pc, **heatmap_args, ax=axes[i],cbar_ax_bbox=[1.05, 0.5, 0.02, 0.3])
            
            sub_ax.set_title(x_i)
            plt.subplots_adjust(wspace=3)


        return fig





    tab_3_0, tab_3_1=  st.tabs(["Plots","Table"])




    with tab_3_0:






        col_3_4a, col_3_4b = st.columns(2,gap='large')

        with col_3_4a:
            metric_options = [i.strip('Test_') for i in st.session_state.train_df.columns if (i.startswith('Test') and not (i.endswith('std')) and not i.endswith('tp)'))]
            metric_selected = st.selectbox("**Select a metric to plot**",metric_options)
        with col_3_4b:   
            groupby_selected = st.selectbox("**Group by:**",["Descriptors","Algorithm"])
            groupby_dict = {"Descriptors": "desc", "Algorithm":"algo"}



        ybar_min = 0.6*np.min([st.session_state.train_df[f'Val_{metric_selected}'].min()-st.session_state.train_df[f'Val_{metric_selected}_std'].max(),
                               st.session_state.train_df[f'Test_{metric_selected}'].min()-st.session_state.train_df[f'Val_{metric_selected}_std'].max()])
        ybar_max = 1.03*np.max([st.session_state.train_df[f'Val_{metric_selected}'].max()+st.session_state.train_df[f'Val_{metric_selected}_std'].max(),
                               st.session_state.train_df[f'Test_{metric_selected}'].max()+st.session_state.train_df[f'Val_{metric_selected}_std'].max()])


        st.write("""##### """)
        st.write("""#####  Evaluate Model Performance """)
        st.write("If the **Cross-validation Scores** are so much better than the **Test Scores**, it might be a sign of **overfitting.**")

        col_3_0a, col_3_0b = st.columns(2,gap='large')
        with col_3_0a:
            st.plotly_chart(plot_barplot(st.session_state.train_df,x=groupby_dict[groupby_selected],y=f'Val_{metric_selected}',xlabel="descriptor",ylabel=f'{metric_selected}',ylim=[ybar_min,ybar_max],title="Cross-validation Scores"),theme='streamlit',use_container_width=True)
        with col_3_0b:
            st.plotly_chart(plot_barplot(st.session_state.train_df,x=groupby_dict[groupby_selected],y=f'Test_{metric_selected}',xlabel="descriptor",ylabel=f'{metric_selected}',ylim=[ybar_min,ybar_max],title="Test Scores"),theme='streamlit',use_container_width=True)

        
        mod_builder.show_advanced_viz = st.checkbox("Show Advanced **Statistical Analysis & Visualization** (Box plot, Sign plot and Scatter plots)")

        if mod_builder.show_advanced_viz:


            #ybar_min,ybar_max=0,1
            st.write("""##### """)
            st.write("""#####  Box Plot of Validation Scores""")
            st.write("If p < 0.05, a **statistically significant** difference exists **within** the group. Otherwise, the observed variations are not significant." )
            st.write("It uses Friedman’s test for k > 2 or Wilcoxon's test for k = 2." )
            st.plotly_chart(plot_boxplot(st.session_state.cv_df,x=groupby_dict[groupby_selected],y=f'test_{metric_selected}',xlabel="descriptor",ylabel=f'{metric_selected}',ylim=[None,None]),theme='streamlit',use_container_width=True)
            
            show_signplot = st.checkbox("Show Pairwise Conover-Friedman Signplot",value=False)
            if show_signplot:
                st.write("""##### """)
                st.write("""#####  Pairwise Conover-Friedman test""")
                st.write("This will tell us which pairs have a significant difference (NS = non-significant)." )

                try:
                    fig = get_signplot(st.session_state.cv_df,x=groupby_dict[groupby_selected],y=f'test_{metric_selected}')
                    st.pyplot(fig)
                except Exception as e:
                    print(e)
                    print("Cannot generate Conover-Friedman plot")
                    



            


        #with tab_3_2:
            def prepare_scatter_regr(train_df,algo_select,desc_select,data_obj):
                
                X_train = data_obj[desc_select].X_train
                X_test = data_obj[desc_select].X_test
                model = train_df[train_df.desc == desc_select][train_df.algo == algo_select ].model_obj.values[0]
                y_train_pred =  model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_train = data_obj[desc_select].y_train
                y_test = data_obj[desc_select].y_test
                df_scatter = pd.DataFrame({"observed": list(y_train) + list(y_test), 
                                        "predicted": list(y_train_pred) + list(y_test_pred),
                                            "set": ["train" for i in range(len(y_train))] + ["test" for i in range(len(y_test))] })


                train_perf = eval_predictions(y_train,y_train_pred,task="Regressor",thresh=None)
                test_perf = eval_predictions(y_test,y_test_pred,task="Regressor",thresh=None)


                return df_scatter, train_perf,test_perf

            def prepare_scatter_clf(train_df,algo_select,desc_select,data_obj):
                
                X_train = data_obj[desc_select].X_train
                X_test = data_obj[desc_select].X_test
                model = train_df[train_df.desc == desc_select][train_df.algo == algo_select ].model_obj.values[0]
                y_train_pred =  model.predict_proba(X_train)[:,1]
                y_test_pred = model.predict_proba(X_test)[:,1]
                y_train = data_obj[desc_select].y_train
                y_test = data_obj[desc_select].y_test
                df_scatter = pd.DataFrame({"observed": list(y_train) + list(y_test), 
                                        "predicted": list(y_train_pred) + list(y_test_pred),
                                            "set": ["train" for i in range(len(y_train))] + ["test" for i in range(len(y_test))] })

                train_perf = eval_predictions(y_train,y_train_pred,task="Classifier",thresh=mod_builder.binary_threshold,y_transformed_binary=True)
                test_perf = eval_predictions(y_test,y_test_pred,task="Classifier",thresh=mod_builder.binary_threshold,y_transformed_binary=True)


                return df_scatter, train_perf,test_perf



            
            
            show_scatter= st.checkbox("Show Scatter Plot",value=False)
            if show_scatter:
            
                col_3_3, col_3_4 = st.columns(2,gap='large')
                with col_3_3:
                    algo_to_plot = st.selectbox("**Select the algorithm to plot**",np.flip(np.unique(st.session_state.train_df.algo)))
                with col_3_4:
                    desc_to_plot = st.selectbox("**Select the descriptor to plot**",np.flip(np.unique(st.session_state.train_df.desc)))     
                    
                

                col_3_5, col_3_6 = st.columns([1,1],gap='large')
                with col_3_5:
                    if task == "Regression":
                        df_scatter,train_perf, test_perf = prepare_scatter_regr(st.session_state.train_df,algo_to_plot,desc_to_plot,st.session_state.data_obj)  
        
                        st.plotly_chart(plot_scatter_regr(df_scatter),key="smiles",on_select="rerun",theme="streamlit", use_container_width=True,height=600)
                    
                    else:
                        
                        df_scatter,train_perf, test_perf = prepare_scatter_clf(st.session_state.train_df,algo_to_plot,desc_to_plot,st.session_state.data_obj) 
                        #st.dataframe(df_scatter)
                    
                        st.plotly_chart(plot_scatter_clf(df_scatter),key="smiles",on_select="rerun",theme="streamlit", use_container_width=True,height=600)


                with col_3_6:
                    st.write("""###""")
                    st.write("**Train Performance**")
                    st.dataframe(train_perf,use_container_width=False)

                    st.write("**Test Performance**")
                    st.dataframe(test_perf,use_container_width=False)

    with tab_3_1:
        st.write("""##### Summary """)
        st.dataframe(st.session_state.train_df[[col for col in st.session_state.train_df.columns if col != 'model_obj']],use_container_width=True)
        st.write("""##### Cross-validation Performance""")
        st.dataframe(st.session_state.cv_df)

    
    mod_builder.proceed_step4 = st.checkbox("Proceed to Save the models")

if mod_builder.proceed_step4:
        
    import joblib
    import pickle
    import base64

    st.markdown('---')
    st.write("""#### Step 4: Save the models for future use """)
    st.write("Select the best combination of algorithm and descriptors based on the analysis above.")


    def save_models(df,desc=desc_choice,algo=algo_choice): #df containing the result of benchmark model function
        mod = df[df['algo'] == algo][df['desc'] == desc].model_obj.values[0]
        return mod
    
    def get_date():
        now = datetime.datetime.now()
        return now.strftime('%d-%m-%y')
    

    col_4_1, col_4_2 = st.columns(2,gap='large')
    with col_4_1:
        algo_to_save = st.selectbox("**Select the algorithm to save**",np.flip(np.unique(st.session_state.train_df.algo)))
    with col_4_2:
        desc_to_save = st.selectbox("**Select the descriptor to save**",np.flip(np.unique(st.session_state.train_df.desc)))   

    output_name = st.text_input("Give a useful filename", "My_Model")


    def download_model(model,output_name):
        #output_model = pickle.dumps(model)
        #b64 = base64.b64encode(output_model).decode()
        #href = f'<a href="data:file/output_model;base64,{b64}" download="{output_name}_{algo_to_save.replace(" ", "")}_{desc_to_save}_{get_date()}.pkl">Download Trained Model .pkl File</a>'
        #st.markdown(href, unsafe_allow_html=True)
        joblib.dump(model,"./temp_model_download.joblib")

    confirm_model_choice = st.checkbox("I confirm my selected model to download.",value=False)    
    if confirm_model_choice:
        try:
            download_model(model=save_models(st.session_state.train_df,desc_to_save,algo_to_save),output_name=output_name)
            with open("./temp_model_download.joblib", "rb") as fp:
                btn = st.download_button(
                    label="Download as **JOBLIB**",
                    data=fp,
                    file_name=f'./{output_name}_{algo_to_save.replace(" ", "")}_{desc_to_save}_{get_date()}.joblib',
                    mime="application/joblib")       
        except Exception as e:
            st.error(e)
            st.error("Failed to save the model. Try to restart training")
            st.button("Restart Training",on_click=restart_session_state)
            

            #restart_session_state()
            #st.rerun()


    #try:
    #    
    #except Exception as e:  
    #    st.write("Model saving failed, trying to restart")      
    #    #restart_session_state()
     #   st.rerun()
      #  st.write(e)
 





################################################################
# Predictor
################################################################

import joblib




class mod_predicter:
    proceed_step2 = False
    proceed_step3 = False
    desc_choice = "ECFP4"
    test_results = pd.DataFrame()
    model_file = None



if mod_builder.objective == "Predict properties":
    st.markdown('---')
    st.write("""#### Step 1: Upload your model (.joblib)""")

    mod_predicter.model_file = st.file_uploader('Select Model File',type=['.pkl','.joblib'],on_change=restart_session_state)

    if mod_predicter.model_file:
        col_5_1, col_5_2 = st.columns(2,gap='medium') 
        with col_5_1:        

            model = joblib.load(mod_predicter.model_file)


            from sklearn.base import is_classifier, is_regressor            

            task_type = "Classifier" if is_classifier(model.named_steps["estimator"]) else "Regressor"

            desc_list = ["ECFP4", "FCFP4", "RDK5","RDKIT"]

            mod_predicter.desc_choice = [desc for desc in desc_list if desc in mod_predicter.model_file.name.split('_')][0]

            if mod_predicter.desc_choice:

                st.write(f"The model is a **{task_type}** using the **{mod_predicter.desc_choice}** descriptor")
            else:
                st.write("Could not automatically identify the molecular features used. Please specify")
                mod_predicter.desc_choice = st.selectbox("Select descriptor used", tuple(["ECFP4", "FCFP4", "RDK5","RDKIT"]))  

            var_pred = st.text_input("Give the name of the variable to be predicted", "My_Property")

            

            
     
            
        mod_predicter.proceed_step2 = st.checkbox("I confirm that the model file and the target variable name is correct.")

if mod_predicter.model_file and mod_predicter.proceed_step2:
    st.markdown('---')
    st.write("""#### Step 2: Upload your CSV file containing SMILES (.csv)""")

    test_file = st.file_uploader('Select CSV File with SMILES',type='.csv',on_change=restart_session_state)
            
    if test_file:

        try:
            df_test = pd.read_csv(test_file) #,engine='python',encoding='unicode_escape'
        except Exception as e:
            st.write(e)

            encoding_list = ['utf_8','ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737'
                            , 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862'
                            , 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950'
                            , 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254'
                            , 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr'
                            , 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2'
                            , 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2'
                            , 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9'
                            , 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab'
                            , 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2'
                            , 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32'
                            , 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8_sig']

            for encoding in encoding_list:
                worked = False
                while worked == False:
                    for encoding in encoding_list:
                        try:
                            df_test = pd.read_csv(test_file, encoding=encoding, nrows=5,engine='python')
                            if df_test is not None:
                                worked = True
                                df_test = pd.read_csv(test_file, encoding=encoding,engine='python')
                        except:
                            worked = False
                            df_test = pd.DataFrame()







                    
        col_6_1, col_6_2 = st.columns(2,gap='medium')    
        with col_6_1:   

            if len(df_test) > 0:

                smi_col_test = st.selectbox("Select SMILES column of the CSV file",tuple(df_test.columns))
                st.session_state.test_file = df_test

                if var_pred in df_test.columns:
                    var_pred = f"{var_pred}_pred" 

                if st.session_state.test_file[smi_col_test].dtype == 'object':
                    #st.write("Checking for invalid SMILES")                
                    canon_smi = []
                    for i,smi in enumerate(st.session_state.test_file[smi_col_test]):
                        try:
                            canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi,sanitize=True))
                            canon_smi.append(canon) 
                        except:
                            canon_smi.append(np.nan)
                            st.write(f"Invalid SMILES {smi} with index {i} was removed")
                    #
                    st.session_state.test_file[smi_col_test] = canon_smi
                    st.session_state.test_file = st.session_state.test_file.dropna(subset=smi_col_test).reset_index(drop=True) 
                    error_in_smiles = False  
                else:
                    st.write("**Error:** Please check that the smiles column contain only SMILES")
                    error_in_smiles = True                    
                    


                test_exp = st.checkbox("I have the true experimental values to compare with the model predictions")

                error_in_y_exp = False
                if test_exp:
                    y_exp = st.selectbox("Select the column with the experimental target value",tuple([i for i in st.session_state.test_file.columns if i != smi_col_test]))

                    if is_numeric_dtype(st.session_state.test_file[y_exp]):
                        st.session_state.test_file[y_exp] = st.session_state.test_file[y_exp].astype(float)
                        
                    else:
                        st.markdown('######')
                        st.write("**Error:** Target variable **must be a number**, if it is categorical, modify the CSV file and convert it to 0 and 1")
                        error_in_y_exp = True 
   
 
                if not error_in_smiles and not error_in_y_exp:
                    mod_predicter.proceed_step3 = st.checkbox("Run Predictions!")    

if mod_predicter.proceed_step3:
    model = joblib.load(mod_predicter.model_file)

    with st.spinner('Calculating Descriptors...'):
        if len(st.session_state.X_desc_test) == 0:
            st.session_state.X_desc_test = get_desc(smi_series=st.session_state.test_file[smi_col_test],desc_name=mod_predicter.desc_choice)

    with st.spinner('Performing Inference...'):
        if var_pred not in st.session_state.test_file.columns:
            try:
                if task_type == "Classifier":
                    st.session_state.test_file[var_pred] = model.predict_proba(st.session_state.X_desc_test)[:,1]
                else:
                    st.session_state.test_file[var_pred] = model.predict(st.session_state.X_desc_test)


                if test_exp:
                    st.session_state.test_file["label"] = [f"exp: {exp:.2f} | pred: {pred:.2f}" for exp,pred in zip(st.session_state.test_file[y_exp],st.session_state.test_file[var_pred]) ]
                else:
                    st.session_state.test_file["label"] = [f"pred: {pred:.2f}" for pred in st.session_state.test_file[var_pred] ] 
            except Exception as e:
                st.write(e)
                st.write("Error. Please check for INVALID SMILES or if the descriptor used was the correct one")

    
    st.markdown('---')
    st.write("""#### Step 3: View and Analyze Results""")

    
    
    st.session_state.test_file["SMILES"] = st.session_state.test_file[smi_col_test]
    
    if task_type == "Classifier":
        st.write("For binary classification, the model predicts the probabilities for the positive class.")
    #st.write("To save the results, click on the **[⋮]** at the upper right")

    tab_6_1, tab_6_2=  st.tabs(["Molecule View", "Table"])
    with tab_6_1:
        raw_html = mols2grid.display(st.session_state.test_file,
                                 subset=["label"] if test_exp else [var_pred],
                                 tooltip=[i for i in st.session_state.test_file.columns.to_list() if i not in ["smiles","SMILES","canon_smi"] ],
                                 smiles_col=smi_col_test,size=(208,104))._repr_html_()
        components.html(raw_html, height=700, scrolling=True) 
    with tab_6_2:
        st.dataframe(st.session_state.test_file)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    
    if len(st.session_state.test_file) > 0:
        st.download_button(
        "**Download CSV**",convert_df(st.session_state.test_file),"file.csv","text/csv",key='download-csv')
     

    if test_exp:
        st.write("""#### Model Performance Statistics""")

        col_7_1, col_7_2 = st.columns(2,gap='large')   
        with col_7_1:
            if task_type == "Classifier":
                thresh_clf = st.slider('Threshold for Positive Class that you have used in model training (Default is median)', st.session_state.test_file[y_exp].min(), st.session_state.test_file[y_exp].max(), st.session_state.test_file[y_exp].median())                
                
                
                y_transformed_binary = True if len(np.unique(st.session_state.test_file[y_exp])) == 2 else False
                perf_stats = eval_predictions(st.session_state.test_file[y_exp],st.session_state.test_file[var_pred],task=task_type,thresh=thresh_clf,y_transformed_binary=y_transformed_binary)
                st.session_state.test_file["exp_class"] = st.session_state.test_file[y_exp] > thresh_clf
                st.plotly_chart(plot_scatter_clf(st.session_state.test_file,x="exp_class",y=var_pred,color="exp_class"))
                

            if task_type == "Regressor":
                st.session_state.test_file["|error|"] = np.abs(st.session_state.test_file[y_exp].values -  st.session_state.test_file[var_pred].values)                
                st.plotly_chart(plot_scatter_regr(st.session_state.test_file,x=y_exp,y=var_pred,color="|error|",color_continuous_scale="Portland"))
                perf_stats = eval_predictions(st.session_state.test_file[y_exp],st.session_state.test_file[var_pred],task=task_type,thresh=None)

        with col_7_2:
            st.write("""##""")
            st.write("""##""")
            st.dataframe(perf_stats) 
     
    #st.write("""#### Step 4: View Results""")


    #model.predict()











  















