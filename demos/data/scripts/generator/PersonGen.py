import pandas as pd
import random
from faker import Faker
from random_address import real_random_address
from phone_gen import PhoneNumber
from datetime import timedelta
from datetime import datetime
import numpy as np
from itertools import count
from names_dataset import NameDataset


class PersonGenerator:

    def __init__(
            self,
            seed: int = 0,
            country: str = 'US',
            people_amt: int = 100,
            affiliations: list = ['Gang Alpha', 'Cartel Beta', 'Gang Gamma', 'Cartel Delta'],
            crimes: list = [
                        "Armed Robbery",
                        "Burglary",
                        "Drug Trafficking",
                        "Vandalism",
                        "Assault",
                        "Money Laundering",
                        "Fraud",
                        "Homicide",
                    ]
            ):

        self.seed = seed
        self.country = country
        self.people_num = people_amt
        Faker.seed(self.seed)
        np.random.seed(self.seed)
        self.fake = Faker()
        self.random = random.Random(self.seed)
        self.phone = PhoneNumber(self.country)
        self.names = NameDataset()
        self.first_names = self.names.get_top_names(n=self.people_num, country_alpha2=self.country)[self.country]
        self.last_names = self.names.get_top_names(n=self.people_num, country_alpha2=self.country, use_first_names=False)[self.country]
        self.address = real_random_address.RandomAddress()
        self.domains = pd.read_csv("domains.txt", header=None)[0].to_list()
        self.affiliations = affiliations
        self.crimes = crimes

    def generate_people(
            self,
            num_records: int = 100,
            min_age: int = 15,
            max_age: int = 85
            ) -> pd.DataFrame:

        records = []
        for _ in range(num_records):
            gender = ["M", "F"]
            sex = self.random.choice(gender)
            record = {
                "first_name": self.random.choice(self.first_names[sex]),
                "last_name": self.random.choice(self.last_names),
                "phone_number": self.phone.get_number(full=False),
                "sex": sex,
                "DOB": self.fake.date_of_birth(
                                        minimum_age=min_age,
                                        maximum_age=max_age
                                        ),
            }
            record["email_address"] = record["first_name"] + record["last_name"] + str(self.random.randint(0, 999)) + self.random.choice(self.domains)
            records.append(record)
        
        df = pd.DataFrame(records)
        return df

    def generate_addresses(
            self,
            num_records: int = 100,
            start_date: str = "-30y",
            end_date: str = "today"
            ) -> pd.DataFrame:

        records = []
        for _ in range(num_records):
            address = self.address()

            record = {
                "address1": address.get('address1', ''),
                "address2": address.get('address2', ''),
                "city": address.get("city", "Unknown City"),
                "date": self.fake.date_between(start_date=start_date, end_date=end_date),
                "state": address.get("state", "Unknown State"),
                "zip": address.get("postalCode", "Unknown PostalCode"),
                "lat": address.get("coordinates", {}).get("lat", 0.0),
                "lon": address.get("coordinates", {}).get("lng", 0.0)
            }
            records.append(record)
        df = pd.DataFrame(records)
        return df

    def generate_call_logs(
            self,
            people_df: pd.DataFrame,
            num_logs: int = 500,
            start_date: str = '-1y'
            ) -> pd.DataFrame:
        
        call_logs = []
        phone_numbers = people_df['phone_number'].tolist()
        
        for _ in range(num_logs):
            caller, callee = self.random.sample(phone_numbers, 2)  # Ensure caller and callee are different
            call_date = self.fake.date_time_between(start_date=start_date)
            call_time = call_date + timedelta(hours=self.random.randint(0, 23), minutes=self.random.randint(0, 59), seconds=self.random.randint(0, 59))
            duration = self.random.randint(1, 3600)  # Call duration in seconds, from 1 sec to 1 hour
            
            call_logs.append({
                "caller": caller,
                "callee": callee,
                "call_date": call_date.strftime('%Y-%m-%d'),
                "call_time": call_time.strftime('%H:%M:%S'),
                "duration_sec": duration
            })
        
        return pd.DataFrame(call_logs)

    def generate_non_affiliated_call_logs(
            self,
            people_df: pd.DataFrame,
            call_logs_df: pd.DataFrame,
            num_calls: int = 500,
            start_date: str = '-1y'
            ) -> pd.DataFrame:
        """
        Generate call logs for non-affiliated individuals, simulating everyday calls.

        :param people_df: DataFrame of people with affiliations.
        :param call_logs_df: DataFrame of call logs to append to.
        :param num_calls: Number of calls to generate among non-affiliated individuals.
        :return: Updated DataFrame with non-affiliated call logs.
        """
        # Filter for non-affiliated individuals
        non_affiliated_people = people_df[people_df['affiliation'] == 'None']
        
        # Generate call logs
        for _ in range(num_calls):
            if len(non_affiliated_people) > 1:
                caller, callee = non_affiliated_people.sample(n=2, replace=False)['phone_number'].values
                self.add_call_log(call_logs_df, caller, callee, start_date)

        return call_logs_df

    def generate_affiliated_call_logs(
            self,
            people_df: pd.DataFrame,
            call_logs_df: pd.DataFrame,
            num_affiliated_calls: int = 100,
            leader_call_percentage: float = 0.05,
            start_date: str = '-1y'
            ) -> pd.DataFrame:
        """
        Generate call logs with a focus on gang affiliations, including both intra-gang and inter-gang communications.
        
        :param people_df: DataFrame of people with affiliations.
        :param call_logs_df: Existing DataFrame of call logs to append to.
        :param num_affiliated_calls: Number of additional affiliated calls to generate.
        :param leader_call_percentage: Percentage of calls that should be between gang leaders (inter-gang calls).
        :return: Updated DataFrame with affiliated call logs.
        """
        affiliated_people = people_df[people_df['affiliation'] != 'None']
        affiliated_groups = affiliated_people['affiliation'].unique()
        
        leader_calls = int(num_affiliated_calls * leader_call_percentage)
        gang_calls = num_affiliated_calls - leader_calls

        # Generate intra-gang calls
        for _ in range(gang_calls):
            gang = self.random.choice(affiliated_groups)
            gang_members = affiliated_people[affiliated_people['affiliation'] == gang]
            
            if len(gang_members) > 1:
                caller, callee = gang_members.sample(n=2, replace=False)['phone_number'].values
                self.add_call_log(call_logs_df, caller, callee, start_date)
        
        # Generate inter-gang calls (leader calls)
        for _ in range(leader_calls):
            gangs = self.random.sample(list(affiliated_groups), 2)
            for gang in gangs:
                gang_leader = affiliated_people[affiliated_people['affiliation'] == gang].sample(n=1)['phone_number'].values[0]
                if gang == gangs[0]:
                    caller = gang_leader
                else:
                    callee = gang_leader
            self.add_call_log(call_logs_df, caller, callee, start_date)

        return call_logs_df

    def add_call_log(
            self,
            call_logs_df: pd.DataFrame,
            caller: str,
            callee: str,
            start_date: str
            ) -> pd.DataFrame:
        """
        Helper function to add a call log entry to the DataFrame.
        """
        call_date = self.fake.date_time_between(start_date=start_date)
        call_time = call_date + timedelta(hours=self.random.randint(0, 23), minutes=self.random.randint(0, 59), seconds=self.random.randint(0, 59))
        duration = self.random.randint(1, 3600)  # Duration in seconds, from 1 sec to 1 hour
        
        new_entry = pd.DataFrame([{
            "caller": caller,
            "callee": callee,
            "call_date": call_date.strftime('%Y-%m-%d'),
            "call_time": call_time.strftime('%H:%M:%S'),
            "duration_sec": duration
        }])

        return pd.concat([call_logs_df, new_entry], ignore_index=True)
    
    def generate_affiliations(
            self,
            people_df: pd.DataFrame,
            percentage_affiliated: float = 0.1,
            lambda_param: float = 1.5
            ) -> pd.DataFrame:
        """
        Generate affiliations for a subset of the provided DataFrame of people.
        
        :param people_df: DataFrame of people.
        :param percentage_affiliated: Approximate percentage of people to have affiliations.
        :param lambda_param: Lambda parameter for the exponential distribution, controlling affiliation spread.
        :return: Updated DataFrame with an 'affiliation' column.
        """
        num_people = len(people_df)
        num_affiliated = int(num_people * percentage_affiliated)
        
        # Determine number of people affiliated with each group, ensuring sum equals num_affiliated
        affiliation_counts = np.random.exponential(lambda_param, len(self.affiliations))
        affiliation_counts = np.round((affiliation_counts / affiliation_counts.sum()) * num_affiliated).astype(int)
        
        # Adjust in case rounding errors cause a mismatch in total counts
        while affiliation_counts.sum() != num_affiliated:
            if affiliation_counts.sum() > num_affiliated:
                affiliation_counts[np.argmax(affiliation_counts)] -= 1
            else:
                affiliation_counts[np.argmin(affiliation_counts)] += 1
        # Assign affiliations to randomly selected people
        people_df['affiliation'] = 'None'
        already_selected = set()
        for count, affiliation in zip(affiliation_counts, self.affiliations):
            eligible_indices = [i for i in range(num_people) if i not in already_selected]
            selected_indices = self.random.sample(eligible_indices, count)
            people_df.loc[selected_indices, 'affiliation'] = affiliation
            already_selected.update(selected_indices)

        return people_df

    def assign_whereabouts_to_people(
            self,
            people_df: pd.DataFrame,
            addresses_df: pd.DataFrame,
            percent_cohabitating: float = 0.2
            ) -> pd.DataFrame:
        # Initially, each person gets a unique address by default (if enough addresses)
        if len(addresses_df) >= len(people_df):
            people_df = pd.concat([people_df, addresses_df.sample(len(people_df)).reset_index(drop=True)], axis=1)
        else:
            raise ValueError("Not enough addresses to assign to each person uniquely.")

        # Identify gang-affiliated individuals for potential cohabitation
        affiliated_groups = people_df[people_df['affiliation'] != 'None']['affiliation'].unique()

        for gang in affiliated_groups:
            gang_members = people_df[people_df['affiliation'] == gang]
            # Decide on how many addresses to group gang members at (e.g., 20% of gang members share addresses)
            num_addresses = int(len(gang_members) * percent_cohabitating)
            shared_addresses = addresses_df.sample(num_addresses)

            for idx, address in shared_addresses.iterrows():
                # Randomly select gang members to live together
                members_to_live_together = gang_members.sample(n=2 if len(gang_members) > 1 else 1)  # At least 2 if possible
                for _, member in members_to_live_together.iterrows():
                    people_df.loc[member.name, ['address1', 'address2', 'city', 'state', 'zip', 'lat', 'lon', 'date']] = address[['address1', 'address2', 'city', 'state', 'zip', 'lat', 'lon', 'date']]

                # Remove the selected members to avoid reselection
                gang_members = gang_members.drop(members_to_live_together.index)

        return people_df
    
    def expand_cases_to_columns(self, people_df: pd.DataFrame) -> pd.DataFrame:
        # Create columns for case details
        max_crimes_per_case = 3  # Adjust based on your dataset
        for i in range(max_crimes_per_case):
            people_df[f'case_number_{i+1}'] = None
            for j in range(max_crimes_per_case):
                people_df[f'crime_{i+1}_{j+1}'] = None

        for index, row in people_df.iterrows():
            for i, case in enumerate(row['cases']):
                if i < max_crimes_per_case:
                    people_df.at[index, f'case_number_{i+1}'] = case['case_number']
                    for j, crime in enumerate(case['crimes']):
                        if j < max_crimes_per_case:
                            people_df.at[index, f'crime_{i+1}_{j+1}'] = crime

        # Drop the original 'cases' column if no longer needed
        # people_df.drop('cases', axis=1, inplace=True)
        
        return people_df

    def generate_and_assign_criminal_records(
            self,
            people_df: pd.DataFrame,
            max_cases_per_person: int = 3
            ) -> pd.DataFrame:
        unique_case_number = count(start=1000, step=1)  # Unique case number generator
        criminal_records = []  # To collect criminal record entries
        gang_related_cases = {}  # To track gang-related case numbers and crimes

        for index, person in people_df.iterrows():
            num_cases = self.random.randint(0, max_cases_per_person)  # Decide how many cases, if any
            records_for_person = {"person_id": index, "cases": []}

            for _ in range(num_cases):
                # Determine if this case is shared (for gang members) or unique
                if person['affiliation'] != 'None' and gang_related_cases.get(person['affiliation']) and self.random.random() < 0.3:
                    # Share an existing case
                    shared_case = self.random.choice(gang_related_cases[person['affiliation']])
                    records_for_person["cases"].append(shared_case)
                else:
                    # Create a new case with 1 or more crimes
                    case_num = next(unique_case_number)
                    crimes_in_case = self.random.sample(self.crimes, self.random.randint(1, min(3, len(self.crimes))))  # Up to 3 crimes per case, adjust as needed
                    new_case = {"case_number": case_num, "crimes": crimes_in_case}
                    records_for_person["cases"].append(new_case)
                    
                    # If gang-affiliated, add this case to the gang's record for potential sharing
                    if person['affiliation'] != 'None':
                        if person['affiliation'] not in gang_related_cases:
                            gang_related_cases[person['affiliation']] = []
                        gang_related_cases[person['affiliation']].append(new_case)

            criminal_records.append(records_for_person)

        # Convert to DataFrame and merge
        criminal_records_df = pd.DataFrame(criminal_records)
        people_df = pd.merge(people_df, criminal_records_df, how='left', left_index=True, right_on='person_id')
        people_df.drop('person_id', axis=1, inplace=True)

        # Handle individuals with no criminal records
        people_df['cases'] = people_df['cases'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [])

        return self.expand_cases_to_columns(people_df)