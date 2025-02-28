import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ace_tools as tools
import re

sns.set()

# import the datasets we will use (ex 1 to 8)
df_hotels = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
)

# add the 'id' column as a unique identifier
df_hotels = df_hotels.reset_index().rename(columns={"index": "id"})

# load the country dataset (ex 9 and 10)
df_country = pd.read_csv(
    "https://gist.githubusercontent.com/tadast/8827699/raw/f5cac3d42d16b78348610fc4ec301e9234f82821/countries_codes_and_coordinates.csv"
)

#
df_country["code"] = df_country["Alpha-3 code"].str.replace('"', "").str.strip()


# ex 1
# ----
def dataframe_summary(df):
    summary = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Null Values": df.isnull().sum(),
            "Null Percentage": (df.isnull().sum() / len(df)) * 100,
            "Unique Values": df.nunique(),
        }
    )
    return summary


summary_df = dataframe_summary(df_hotels)
print(summary_df)


# ex 2
# ----
def cancellation_summary(df, cancellation_column):
    cancellation_counts = df[cancellation_column].value_counts()
    total = cancellation_counts.sum()
    cancellation_percentages = (cancellation_counts / total) * 100
    summary = pd.DataFrame(
        {"Count": cancellation_counts, "Percentage": cancellation_percentages}
    )
    return summary


summary_df = cancellation_summary(df_hotels, "is_canceled")
print(summary_df)


# ex 3
# ----
def hotel_cancellation_rate(df, hotel_column, cancellation_column):
    cancellation_summary = df.groupby(hotel_column)[cancellation_column].mean() * 100
    return cancellation_summary


cancellation_rates = hotel_cancellation_rate(df_hotels, "hotel", "is_canceled")
print(cancellation_rates)

most_canceled_hotel = cancellation_rates.idxmax()
highest_percentage = cancellation_rates.max()
print(
    f"The hotel type with the highest cancellation rate is: {most_canceled_hotel} with {highest_percentage:.2f}% cancellations."
)

# ex 4
# ----
df_checkout = df_hotels[df_hotels["is_canceled"] == 0]
print(df_checkout.head())

# ex 5
# ----
monthly_reservations = (
    df_checkout.groupby(["hotel", "arrival_date_month"]).size().unstack()
)
print(monthly_reservations)

most_reservations = monthly_reservations.idxmax(axis=1)
print("\nMonth with the highest number of reservations for each hotel type:")
print(most_reservations)

month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

df_checkout["arrival_date_month_num"] = df_checkout["arrival_date_month"].map(
    month_mapping
)
monthly_reservations_num = (
    df_checkout.groupby(["hotel", "arrival_date_month_num"]).size().unstack()
)
print("\nReservations per month (numeric format):")
print(monthly_reservations_num)

monthly_reservations.T.plot(kind="bar", figsize=(12, 6), alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Number of Reservations")
plt.title("Reservations per Month for Each Hotel Type")
plt.legend(title="Hotel Type")
plt.xticks(rotation=45)
plt.show()


# ex 6
# ----
month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

df_checkout["arrival_date_month_num"] = df_checkout["arrival_date_month"].map(
    month_mapping
)
df_checkout["arrival_date"] = pd.to_datetime(
    df_checkout[
        ["arrival_date_year", "arrival_date_month_num", "arrival_date_day_of_month"]
    ]
)
print(
    df_checkout[
        [
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_day_of_month",
            "arrival_date",
        ]
    ].head()
)


# ex 7
# ----
df_daily_reservation = (
    df_checkout.groupby("arrival_date").size().reset_index(name="num_reservations")
)
print(df_daily_reservation.head())

tools.display_dataframe_to_user(
    name="Daily Reservations", dataframe=df_daily_reservation
)


# ex 8
# ----
adr_summary = (
    df_checkout.groupby(["hotel", "customer_type"])["adr"].mean().reset_index()
)
highest_adr_per_hotel = adr_summary.loc[adr_summary.groupby("hotel")["adr"].idxmax()]
print("Average ADR per hotel and customer type:")
print(adr_summary)
print("\nCustomer type with highest ADR in each hotel:")
print(highest_adr_per_hotel)

tools.display_dataframe_to_user(name="ADR Summary", dataframe=adr_summary)
tools.display_dataframe_to_user(
    name="Highest ADR per Hotel", dataframe=highest_adr_per_hotel
)


# ex 9
# ----
df_merged = df_checkout.merge(
    df_country, left_on="country", right_on="code", how="left"
)
country_reservations = (
    df_merged.groupby("country_name").size().reset_index(name="num_reservations")
)

top_10_countries = country_reservations.sort_values(
    by="num_reservations", ascending=False
).head(10)
print("Top 10 countries with the largest number of reservations:")
print(top_10_countries)

tools.display_dataframe_to_user(name="Top 10 Countries", dataframe=top_10_countries)


# ex 10
# ----
df_checkout["total_guests"] = (
    df_checkout["adults"]
    + df_checkout["children"].fillna(0)
    + df_checkout["babies"].fillna(0)
)
average_guests = df_checkout["total_guests"].mean()
max_guests = df_checkout["total_guests"].max()
max_guests_row = df_checkout[df_checkout["total_guests"] == max_guests]

print(f"Average number of guests per reservation: {average_guests:.2f}")
print(f"Highest number of guests in a single reservation: {max_guests}")
print("\nReservation with the highest number of guests:")
print(max_guests_row)

tools.display_dataframe_to_user(
    name="Reservation with Highest Guests", dataframe=max_guests_row
)
