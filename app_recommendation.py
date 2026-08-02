import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Zomato Recommendation System",page_icon="🍽️",layout="wide")

zomato=pd.read_csv("zomato.csv",encoding="latin1")

model=joblib.load("zomato_rating_model.pkl")
feature_names=joblib.load("feature_names.pkl")
city_country=joblib.load("city_country.pkl")

st.title("🍽️ Zomato Restaurant Recommendation & Rating Prediction")
st.write("Search restaurants and compare the actual rating with the ML predicted rating.")

city=st.sidebar.selectbox("📍 City",sorted(zomato["City"].dropna().unique()))

cuisine_list=sorted(zomato["Cuisines"].dropna().str.split(",").explode().str.strip().unique())
cuisine=st.sidebar.selectbox("🍴 Cuisine",cuisine_list)

budget=st.sidebar.slider("💰 Maximum Budget for Two (₹)",0,10000,1000,100)

if st.sidebar.button("🔍 Find Restaurants"):
    result=zomato[
        (zomato["City"]==city) &
        (zomato["Average Cost for two"]<=budget) &
        (zomato["Cuisines"].fillna("").str.contains(cuisine,case=False))
    ].copy()

    if result.empty:
        st.warning("No matching restaurants found.")
    else:
        names=result["Restaurant Name"].tolist()
        selected=st.selectbox("Select Restaurant",names)

        row=result[result["Restaurant Name"]==selected].iloc[0]
        
        st.info("### 🍽️ Restaurant Details")
  
        c1,c2=st.columns(2)

        with c1:
            st.markdown(f"## 🍽️ {row['Restaurant Name']}")
            st.write(f"**Address:** {row['Address']}")
            st.write(f"**City:** {row['City']}")
            st.write(f"**Country:** {city_country.get(row['City'],'Unknown')}")
            st.write(f"**Cuisine:** {row['Cuisines']}")

        with c2:
            st.write(f"**Average Cost:** ₹{row['Average Cost for two']}")
            st.write(f"**Votes:** {row['Votes']}")
            st.write(f"**Table Booking:** {row['Has Table booking']}")
            st.write(f"**Online Delivery:** {row['Has Online delivery']}")
            st.write(f"**Delivering Now:** {row['Is delivering now']}")

        st.divider()

        input_data=pd.DataFrame(0,index=[0],columns=feature_names)

        if "Average Cost for two" in input_data.columns:
            input_data["Average Cost for two"]=row["Average Cost for two"]
        if "Price range" in input_data.columns:
            input_data["Price range"]=row["Price range"]
        if "Votes" in input_data.columns:
            input_data["Votes"]=row["Votes"]

        mappings={
            "City":"City_",
            "Country":"Country_",
            "Currency":"Currency_",
            "Has Table booking":"Has Table booking_",
            "Has Online delivery":"Has Online delivery_",
            "Is delivering now":"Is delivering now_",
            "Switch to order menu":"Switch to order menu_"
        }

        for col,prefix in mappings.items():
            value=row[col] if col!="Country" else city_country.get(row["City"],"Unknown")
            feature=f"{prefix}{value}"
            if feature in input_data.columns:
                input_data[feature]=1

        prediction=float(model.predict(input_data)[0])

        actual=float(row["Aggregate rating"])

        st.subheader("⭐ Rating Comparison")

        a,b=st.columns(2)
        a.metric("⭐ Actual Rating", f"{actual:.2f}/5")
        b.metric("🤖 Predicted Rating", f"{prediction:.2f}/5")
        st.subheader("Restaurant Rating")

        stars = "⭐" * int(round(actual))
        st.markdown(f"### {stars}")

        st.progress(actual / 5)
       

        st.progress(min(prediction/5,1.0))

        diff=abs(prediction-actual)
        if diff<0.25:
            st.success("✅ Excellent prediction")
        elif diff<0.5:
            st.info("👍 Good prediction")
        else:
            st.warning("Prediction differs from actual rating.")

        lat=row["Latitude"]
        lon=row["Longitude"]
        st.link_button("📍 Open in Google Maps",f"https://www.google.com/maps?q={lat},{lon}")

        st.markdown("## 🏆 You may also like")
        similar=result[result["Restaurant Name"]!=selected].sort_values(
            ["Aggregate rating","Votes"],ascending=False
        )[["Restaurant Name","Aggregate rating","Average Cost for two"]].head(5)
        st.dataframe(similar,use_container_width=True)
