import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set()


# "Global" definitions
# --------------------


df_hotels = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/"
    "2020/2020-02-11/hotels.csv"
)
df_country = pd.read_csv(
    "https://gist.githubusercontent.com/tadast/8827699/raw/"
    "f5cac3d42d16b78348610fc4ec301e9234f82821/countries_codes_and_coordinates.csv"
)
df_hotels = df_hotels.reset_index().rename(columns={"index": "id"})
df_country["code"] = df_country["Alpha-3 code"].str.replace('"', "").str.strip()


# Functions
# ---------


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


def cancellation_summary(df, cancellation_column):
    cancellation_counts = df[cancellation_column].value_counts()
    total = cancellation_counts.sum()
    cancellation_percentages = (cancellation_counts / total) * 100
    summary = pd.DataFrame(
        {"Count": cancellation_counts, "Percentage": cancellation_percentages}
    )
    return summary


def hotel_cancellation_rate(df, hotel_column, cancellation_column):
    cancellation_summary = df.groupby(hotel_column)[cancellation_column].mean() * 100
    return cancellation_summary


if __name__ == "__main__":

    # Exercise 1
    # ----------
    # Create a function with one argument formed in DataFrame to check
    # the data type, the number of null values, the percentage of null values
    # and the number of unique values for each column.

    print("\n\nExercise 1")
    print("----------\n")
    summary_df = dataframe_summary(df_hotels)
    print(summary_df)

    # Exercise 2
    # ----------
    # How many visitors are there who cancel the reservation and who don’t?
    # And from that number draw conclusions about the proportions of each.

    print("\n\nExercise 2")
    print("----------\n")
    summary_df = cancellation_summary(df_hotels, "is_canceled")
    print(summary_df)

    # Exercise 3
    # ----------
    # -> For “City Hotel”, what is the percentage of canceled reservations?
    # -> For “Resort Hotel”, what is the percentage of canceled reservations?
    # -> What type of hotel that has the bigger percentage of canceled reservations?

    print("\n\nExercise 3")
    print("----------\n")
    cancellation_rates = hotel_cancellation_rate(df_hotels, "hotel", "is_canceled")
    print(cancellation_rates)
    most_canceled_hotel = cancellation_rates.idxmax()
    highest_percentage = cancellation_rates.max()
    print(
        f"The hotel type with the highest cancellation rate is: {most_canceled_hotel} "
        f"with {highest_percentage:.2f}% cancellations."
    )

    # Exercise 4
    # ----------
    # Filter data so that it only displays the visitors who don’t cancel
    # the reservation and save the result in df_checkout variable.

    print("\n\nExercise 4")
    print("----------\n")
    df_checkout = df_hotels[df_hotels["is_canceled"] == 0].copy()
    print(df_checkout.head())

    # Exercise 5
    # ----------
    # -> Show the number of reservations per month of arrival for each type of hotel!
    # -> Then in which month there are the most reservations in each type of hotel?
    #    Make a conclusion whether the trend is the same in both types of hotels?
    # -> Do as the previous point but with the name of the month that has been mapped into months in numbers.
    # Note: for this and subsequent questions will use the `df_checkout` dataframe.

    print("\n\nExercise 5")
    print("----------\n")
    monthly_reservations = (
        df_checkout.groupby(["hotel", "arrival_date_month"]).size().unstack()
    )
    print(monthly_reservations)
    most_reservations = monthly_reservations.idxmax(axis=1)
    print(
        f"\nMonth with the highest number of reservations for each hotel type: {most_reservations}"
    )
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
    print(f"\nReservations per month (numeric format): {monthly_reservations_num}")
    monthly_reservations.T.plot(kind="bar", figsize=(12, 6), alpha=0.7)
    plt.xlabel("Month")
    plt.ylabel("Number of Reservations")
    plt.title("Reservations per Month for Each Hotel Type")
    plt.legend(title="Hotel Type")
    plt.xticks(rotation=45)
    plt.show()

    # Exercise 6
    # ----------
    # -> Create a new column named arrival_date which contains complete
    #    information about the year, month, and date of arrival (example: 2022-12-27)
    # -> Change the column to datetime type

    print("\n\nExercise 6")
    print("----------\n")
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
    df_checkout.loc[:, "arrival_date_month_num"] = df_checkout[
        "arrival_date_month"
    ].map(month_mapping)
    df_checkout["arrival_date"] = pd.to_datetime(
        df_checkout[
            ["arrival_date_year", "arrival_date_month_num", "arrival_date_day_of_month"]
        ].rename(
            columns={
                "arrival_date_day_of_month": "day",
                "arrival_date_month_num": "month",
                "arrival_date_year": "year",
            }
        )
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

    # Exercise 7
    # ----------
    # Create a dataframe containing df_daily_reservation.

    print("\n\nExercise 7")
    print("----------\n")
    df_daily_reservation = (
        df_checkout.groupby("arrival_date").size().reset_index(name="num_reservations")
    )
    print(df_daily_reservation.head())

    # Exercise 8
    # ----------
    # -> What is the average ADR (Average Daily Rate) based on hotel type and customer type?
    # -> Which type of customer has the highest the average of ADR in each type of hotel?

    print("\n\nExercise 8")
    print("----------\n")
    adr_summary = (
        df_checkout.groupby(["hotel", "customer_type"])["adr"].mean().reset_index()
    )
    highest_adr_per_hotel = adr_summary.loc[
        adr_summary.groupby("hotel")["adr"].idxmax()
    ]
    print("Average ADR per hotel and customer type:")
    print(adr_summary)
    print("\nCustomer type with highest ADR in each hotel:")
    print(highest_adr_per_hotel)

    # Exercise 9
    # ----------
    # By using the `df_country` dataframe which contains the country name and country code information,
    # show the 10 countries with the largest number of reservations.
    # (You need to combine this dataframe with another right one)

    print("\n\nExercise 9")
    print("----------\n")
    df_country.rename(columns={"Alpha-3 code": "country"}, inplace=True)
    df_merged = df_checkout.merge(df_country, on="country", how="left")
    country_reservations = (
        df_merged.groupby("country").size().reset_index(name="num_reservations")
    )
    top_10_countries = country_reservations.sort_values(
        by="num_reservations", ascending=False
    ).head(10)
    print("Top 10 countries with the largest number of reservations:")
    print(top_10_countries)

    # Exercise 10
    # -----------
    # -> How many average guests stay for each reservation?
    # -> Based on the dataset, what is the highest number of guests?
    #    Also show the reservation data row that has the highest number of guests.

    print("\n\nExercise 10")
    print("-----------\n")
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
