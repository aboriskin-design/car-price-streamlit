import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


@st.cache_resource
def load_artifact(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_input(df, artifact):
    df = df.copy()

    # если прилетел name - делаю brand
    if "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.split().str[0].str.lower()
        df = df.drop(columns=["name"])

    # если brand вообще нет - добавлю как unknown
    if "brand" not in df.columns:
        df["brand"] = "unknown"

    if "selling_price" in df.columns:
        df = df.drop(columns=["selling_price"])

    # привожу колонки, которые ожидаем
    for c in artifact["fill_num"]:
        if c not in df.columns:
            df[c] = np.nan

    for c in artifact["fill_cat"]:
        if c not in df.columns:
            df[c] = np.nan

    # заполняю пропуски
    for c, v in artifact["fill_num"].items():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(v)

    for c, v in artifact["fill_cat"].items():
        df[c] = df[c].fillna(v).astype(str)

    # one-hot
    df_ohe = pd.get_dummies(df, columns=artifact["cols_to_ohe"], drop_first=True)

    # выравниваю колонки под train
    X = df_ohe.reindex(columns=artifact["ohe_columns"], fill_value=0)

    return X


def plot_eda(df):
    st.write("Пара простых графиков по данным")

    num_cols = [c for c in df.columns if c in ["year", "km_driven", "mileage", "engine", "max_power", "selling_price"]]
    for c in num_cols:
        fig, ax = plt.subplots()
        ax.hist(pd.to_numeric(df[c], errors="coerce").dropna(), bins=30)
        ax.set_title(c)
        st.pyplot(fig)


def plot_coefs(artifact, top_n=15):
    model = artifact["model"]
    coefs = pd.Series(model.coef_, index=artifact["ohe_columns"])
    coefs_abs = coefs.abs().sort_values(ascending=False).head(top_n)
    coefs_top = coefs.loc[coefs_abs.index]

    fig, ax = plt.subplots()
    coefs_top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Топ коэффициентов Ridge (по модулю)")
    st.pyplot(fig)


st.set_page_config(page_title="Car price app", layout="wide")
st.title("Прогноз цены авто")

artifact = load_artifact("model.pkl")

tab1, tab2, tab3 = st.tabs(["EDA", "Predict", "Model weights"])

with tab1:
    st.subheader("EDA")
    st.write("Можно загрузить csv и посмотреть базовые графики. Если не загрузить, попробую открыть data/train.csv")

    uploaded = st.file_uploader("Загрузить csv для EDA", type=["csv"])
    if uploaded is not None:
        df_eda = pd.read_csv(uploaded)
        st.dataframe(df_eda.head(10))
        plot_eda(df_eda)
    else:
        try:
            df_eda = pd.read_csv("data/train.csv")
            st.dataframe(df_eda.head(10))
            plot_eda(df_eda)
        except:
            st.info("Нет data/train.csv, загрузи файл для EDA вручную")

with tab2:
    st.subheader("Predict")

    mode = st.radio("Как подать данные", ["Загрузить CSV", "Ввести вручную"])

    if mode == "Загрузить CSV":
        up = st.file_uploader("CSV с признаками", type=["csv"], key="pred_csv")
        if up is not None:
            df_in = pd.read_csv(up)
            st.dataframe(df_in.head(10))

            X = preprocess_input(df_in, artifact)
            preds = artifact["model"].predict(X)

            out = df_in.copy()
            out["pred_price"] = preds
            st.dataframe(out.head(20))

            st.download_button(
                "Скачать результат csv",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input("year", value=2015)
            km_driven = st.number_input("km_driven", value=50000)
            mileage = st.number_input("mileage", value=20.0)

        with col2:
            engine = st.number_input("engine", value=1200)
            max_power = st.number_input("max_power", value=80.0)
            seats = st.selectbox("seats", [4, 5, 6, 7, 8, 9, 10], index=1)

        with col3:
            fuel = st.selectbox("fuel", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("transmission", ["Manual", "Automatic"])
            owner = st.selectbox("owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
            brand = st.text_input("brand", value="maruti")

        df_in = pd.DataFrame([{
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "brand": brand
        }])

        X = preprocess_input(df_in, artifact)
        pred = artifact["model"].predict(X)[0]
        st.write("Предсказанная цена:", float(pred))

with tab3:
    st.subheader("Веса модели")
    plot_coefs(artifact, top_n=20)
