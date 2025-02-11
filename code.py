import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

st.set_page_config(page_title="Crime Rate Prediction", layout="wide")

st.title("Crime Rate Prediction")
st.subheader("Predicts your data in an accurate manner:")

with st.container():
    st.write("---")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Crime Rate Analysis", "Prediction"],
        icons=["house", "award-fill", "calendar"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    st.subheader('Upload the dataset either in excel/csv format:')
    uploaded_file = st.file_uploader("-->", type=['csv', 'xlsx'])

    if uploaded_file:
        st.markdown('---')
        try:
            df = pd.read_csv(uploaded_file)  # Assuming it's a CSV
        except pd.errors.ParserError:
            df = pd.read_excel(uploaded_file) # Or it could be an Excel file

        if selected == "Home":
            st.subheader('Process of predicting the crime rate')
            st.write(
                "Crime analysis and prediction is a systematic approach for identifying the crime."
                " This system can predict regions which have high probability for crime occurrences and visualize crime "
                "prone areas."
                " Using the concept of data mining we can extract previously unknown, useful information from an "
                "unstructured data."
                " We propose a system which can analyze, detect, and predict various crime probability in a given region."
            )
            st.dataframe(df)
            st.write(
                "[Reference](https://www.kaggle.com/code/sugandhkhobragade/caste-crimes-crimes-against-sc-2001-2013)"
            )

        elif selected == "Crime Rate Analysis":
            filtered_df = dataframe_explorer(df)
            st.dataframe(filtered_df, use_container_width=True)
            st.write("---")
            st.subheader("State Wise Total IPC Crimes:")
            st.dataframe(df.groupby(by=["STATE/UT"]).sum()[["TOTAL IPC CRIMES"]])

            fig = px.box(df, x="YEAR", y="DISTRICT", title="District Analysis by Year")  # more meaningful title
            st.plotly_chart(fig)

            st.subheader("Analysis of Datasets by Graphs:")
            if st.button('Analysis'):
                st.subheader('Bar Chart:')
                st.bar_chart(df.groupby("YEAR")["MURDER"].sum()) # or a different aggregate based on what makes sense

                st.subheader('Plotly Chart:')
                fig_plotly = px.line(df.groupby("YEAR")["MURDER"].sum(), title="Murder Rate Over Years") # line chart may make more sense.
                st.plotly_chart(fig_plotly)

                st.subheader('Area Chart:')
                st.area_chart(df.groupby("YEAR")["MURDER"].sum())

            if st.checkbox("Area graph with different factors"):
                all_columns = df.columns.to_list()
                feat_choices = st.multiselect("Choose a Feature", all_columns)
                if feat_choices:
                    new_df = df[feat_choices]
                    st.area_chart(new_df)
                else:
                    st.warning("Please select at least one feature for the Area Chart.")

        elif selected == "Prediction":
            st.dataframe(df)
            st.subheader("Train the data:")
            trainmodel = st.checkbox("Train model")

            if trainmodel:
                st.header("TRAINING THE DATA:")

                try:
                    y = df.YEAR
                    X = df[["MURDER"]].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Added random state. set test size to 20%
                    lrgr = LinearRegression()
                    lrgr.fit(X_train, y_train)
                    pred = lrgr.predict(X_test)
                    mse = mean_squared_error(y_test, pred)
                    rmse = sqrt(mse)
                    r2 = r2_score(y_test,y_pred)

                    st.markdown(f"""
                    Linear Regression model trained:
                    - MSE: {mse:.2f}
                    - RMSE: {rmse:.2f}
                    - R2 Score: {r2:.2f} #Added R2 Score
                    """)
                    st.success('Model trained successfully')
                    X = df[["MURDER"]].values  #Reshape the data

                    kf = KFold(n_splits=10, shuffle=True, random_state=42) #Added shuffling and randomstate.
                    mse_list = []
                    rmse_list = []
                    r2_list = []
                    idx = 1

                    fig, ax = plt.subplots() # use subplots to plot nicely in streamlit
                    my_bar = st.progress(0)

                    for i, (train_index, test_index) in enumerate(kf.split(X)): #Enumerate helps with the progress bar
                        my_bar.progress((i + 1) * 10)
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        lrgr = LinearRegression()
                        lrgr.fit(X_train, y_train)
                        pred = lrgr.predict(X_test)
                        mse = mean_squared_error(y_test, pred)
                        rmse = sqrt(mse)
                        r2 = r2_score(y_test, pred)

                        mse_list.append(mse)
                        rmse_list.append(rmse)
                        r2_list.append(r2)
                        ax.plot(y_test, pred, '.', label=f"Fold {idx}")  # Scatter plot
                        idx += 1

                    ax.legend() # Moved legend here to not overlap
                    ax.set_xlabel("Actual MURDER Value")
                    ax.set_ylabel("Predicted MURDER Value")
                    ax.set_title("Actual vs Predicted Values Across Folds")
                    st.pyplot(fig)
                    st.subheader('Average Metrics Across Folds:')

                    st.write(f"- Average MSE: {np.mean(mse_list):.2f}")  #Formatted the value to 2 decimal places.
                    st.write(f"- Average RMSE: {np.mean(rmse_list):.2f}")
                    st.write(f"- Average R2 Score: {np.mean(r2_list):.2f}")


                except AttributeError as e:
                    st.error(f"An AttributeError occurred: {e}.  Make sure that 'YEAR' and 'MURDER' are column names in your dataset.")
                except KeyError as e:
                    st.error(f"A KeyError occurred: {e}. Please verify your dataset contains column names 'YEAR' and 'MURDER'.")

            st.subheader('District Prediction:')
            row1 = st.selectbox('Select the district:', df['DISTRICT'].unique()) #unique() so you have distict select options
            row2 = st.selectbox('Select the state:', df['STATE/UT'].unique())

            selected_rows = df[(df['DISTRICT'] == row1) & (df['STATE/UT'] == row2)]

            if st.button('Predict'):
                if not selected_rows.empty:
                    result = selected_rows['MURDER'].sum()
                    st.write("The total predicted murders: " + str(result))

                    if result > 1000:
                        st.subheader('This area might have a high Crime Rate')
                    else:
                        st.subheader('This area has a lower Crime Rate')
                else:
                    st.error("No data found for the selected District and State. Please select other values.")

            if st.checkbox("Area graph with different factors for predictions"):
                all_columns = df.columns.to_list()
                feat_choices = st.multiselect("Choose a Feature", all_columns)
                if feat_choices:
                    new_df = df[feat_choices]
                    st.subheader('Area Chart:')
                    st.area_chart(new_df)
                else:
                    st.warning("Please select at least one feature for the Area Chart.")
