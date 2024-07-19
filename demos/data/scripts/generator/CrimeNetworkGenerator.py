import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import factory.random
from datetime import datetime, timedelta
from ProfileGenerator import ProfileFactory
from scipy.spatial import cKDTree
from itertools import count
import graphistry


class PersonNetworkGenerator:
    def __init__(
            self,
            n_kingpins: int = 4,
            dealers_per_kingpin: int = 5,
            users_per_dealer: int = 3,
            dealer_normal_connections: int = 4,
            kingpin_normal_connections: int = 3,
            within_group_connections: int = 4,
            random_connections: int = 3,
            max_calls_per_edge: int = 11,
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
                    ],
            max_crimes_per_case: int = 3,
            max_cases_per_person: int = 3,
            n_normal: int = 1000,
            postal_code: int = None,
            state: str = None,
            call_start_date: str = "2022-1-1",
            call_end_date: str = "2023-12-31",
            max_num_whereabouts: int = 4,
            leader_to_leader_call_chance: float = 0.05,
            shared_case_percentage: float = 0.3,
            ):

        self.n_kingpins = n_kingpins
        self.dealers_per_kingpin = dealers_per_kingpin
        self.users_per_dealer = users_per_dealer
        self.dealer_normal_connections = dealer_normal_connections
        self.kingpin_normal_connections = kingpin_normal_connections
        self.within_group_connections = within_group_connections
        self.random_connections = random_connections
        self.n_normal = n_normal
        self.node_df = None
        self.edge_df = None
        self.labels = None
        self.seed = 42
        np.random.seed(self.seed)
        factory.random.reseed_random(self.seed)
        self.postal_code = postal_code
        self.state = state
        self.affiliations = affiliations
        self.crimes = crimes
        self.max_crimes_per_case = max_crimes_per_case
        self.max_cases_per_person = max_cases_per_person
        self.max_calls_per_edge = max_calls_per_edge
        self.call_start_date = call_start_date
        self.call_end_date = call_end_date
        self.max_num_whereabouts = max_num_whereabouts
        self.leader_to_leader_call_chance = leader_to_leader_call_chance
        self.shared_case_percentage = shared_case_percentage

    #NETWORK GENERATION
    def generate_network(self):
        # Generate clusters for kingpins, dealers, and users
        X_kingpins, _ = make_blobs(
            n_samples=self.n_kingpins,
            centers=self.n_kingpins,
            cluster_std=1.0,
            random_state=self.seed
            )

        X_dealers, _ = make_blobs(
            n_samples=self.dealers_per_kingpin * self.n_kingpins,
            centers=X_kingpins,
            cluster_std=2.5,
            random_state=self.seed
            )

        X_users, _ = make_blobs(
            n_samples=self.users_per_dealer * self.dealers_per_kingpin * self.n_kingpins,
            centers=X_dealers,
            cluster_std=3.5,
            random_state=self.seed
            )

        X_normal = np.random.rand(self.n_normal, 2) * 100  # Normal people data

        # Combine all data
        points = np.vstack([X_kingpins, X_dealers, X_users, X_normal])
        self.labels = ['kingpin']*self.n_kingpins + \
                      ['dealer']*self.dealers_per_kingpin*self.n_kingpins + \
                      ['user']*self.dealers_per_kingpin*self.n_kingpins*self.users_per_dealer + \
                      ['normal']*self.n_normal

        # Create DataFrame for nodes
        self.node_df = pd.DataFrame(points, columns=['x', 'y'])
        self.node_df['node_id'] = range(len(self.node_df))
        self.node_df['type'] = self.labels

        # Assign personal details
        self.assign_personal_details(self.postal_code, self.state, self.max_num_whereabouts)

        # assign affiliations
        self.assign_affiliations()

        # Generate criminal records
        self.generate_and_assign_criminal_records()

        # Generate edges
        self.generate_edges()

        # Generate call logs
        self.generate_and_assign_call_logs(self.call_start_date, self.call_end_date)

    def calculate_nearest_kingpin(self):
        # Extract coordinates for kingpins and dealers
        kingpin_coords = self.node_df[self.node_df['type'] == 'kingpin'][['x', 'y']].to_numpy()
        dealer_coords = self.node_df[self.node_df['type'] == 'dealer'][['x', 'y']].to_numpy()

        # Find nearest kingpin index for each dealer
        nearest_kingpin_indices = self.find_nearest_kingpin_index(dealer_coords, kingpin_coords)

        # Map nearest kingpin indices back to the original DataFrame indices of kingpins
        kingpin_df_indices = self.node_df[self.node_df['type'] == 'kingpin'].index.to_numpy()
        mapped_kingpin_indices = kingpin_df_indices[nearest_kingpin_indices]

        # Assign the mapped kingpin indices to dealers in the DataFrame
        self.node_df.loc[self.node_df['type'] == 'dealer', 'nearest_kingpin_index'] = mapped_kingpin_indices

    def find_nearest_kingpin_index(
            self,
            dealer_coords: np.array,
            kingpin_coords: np.array
            ) -> np.array:
        # Create a KD-tree for kingpin locations
        tree = cKDTree(kingpin_coords)

        # Query the tree for the nearest kingpin to each dealer
        # 'query' returns a tuple where the first element is the distance
        # and the second element is the index of the nearest kingpin in the tree
        _, nearest_kingpin_indices = tree.query(dealer_coords, k=1)

        return nearest_kingpin_indices

    def ensure_kingpin_dealer_connectivity(self) -> list:
        edge_list = []
        kingpins = self.node_df[self.node_df['type'] == 'kingpin']
        for kingpin_index in kingpins.index:
            affiliated_dealers = self.node_df[(self.node_df['type'] == 'dealer') & (self.node_df['affiliation'] == self.node_df.at[kingpin_index, 'affiliation'])].index
            # Ensure each kingpin has connections to dealers
            if not affiliated_dealers.empty:
                selected_dealers = np.random.choice(affiliated_dealers, size=min(3, len(affiliated_dealers)), replace=False)
                for dealer_index in selected_dealers:
                    edge_list.append((kingpin_index, dealer_index))
        return edge_list

    def connect_dealers_to_users(self) -> list:
        edge_list = []
        dealers = self.node_df[self.node_df['type'] == 'dealer']
        users = self.node_df[self.node_df['type'] == 'user'].index
        for dealer_index in dealers.index:
            # Select a random number of users to connect with each dealer
            num_connections = self.users_per_dealer  # For example, each dealer connects with 2 to 4 users
            selected_users = np.random.choice(users, size=num_connections, replace=False)
            for user_index in selected_users:
                edge_list.append((dealer_index, user_index))
        return edge_list

    def connect_within_group(self) -> list:
        # Exclude kingpins and normal individuals for within-group connections
        group_nodes = self.node_df[~self.node_df['type'].isin(['kingpin', 'normal'])]

        # Group by affiliation and type
        grouped = group_nodes.groupby(['affiliation', 'type'])

        # Initialize an empty list to store edges
        edge_list = []

        # Iterate over each group
        for name, group in grouped:
            # Generate connections for each node in the group
            for node_index in group.index:
                # Identify potential connections within the same affiliation and type
                potential_connections = group.index[group.index != node_index]
                # Randomly select a subset for connections
                num_connections = np.random.randint(1, self.within_group_connections)  # Adjust numbers as needed
                if not potential_connections.empty:
                    selected_connections = np.random.choice(potential_connections, size=min(len(potential_connections), num_connections), replace=False)
                    # Add connections to the edge list
                    edge_list.extend([(node_index, connection) for connection in selected_connections])

        return edge_list

    def connect_randomly(self) -> list:
        # Decide randomly if a node should form random connections
        nodes_to_connect = self.node_df.index[np.random.rand(len(self.node_df)) < 0.1]

        # Function to generate random connections for a node
        def generate_random_connections(node):
            # Exclude self-connections
            potential_connections = self.node_df.index[self.node_df.index != node]
            num_connections = np.random.randint(1, self.random_connections)  # Adjust numbers as needed
            selected_connections = np.random.choice(potential_connections, size=min(len(potential_connections), num_connections), replace=False)
            return [(node, connection) for connection in selected_connections]

        # Generate random connections for each selected node
        edge_list = [edge for node in nodes_to_connect for edge in generate_random_connections(node)]

        return edge_list

    def connect_to_normals(self) -> list:
        # Define which roles should have connections to normal individuals
        roles_with_normal_connections = ['kingpin', 'dealer']

        # Filter the DataFrame for normal individuals
        normal_people = self.node_df[self.node_df['type'] == 'normal'].index

        # Filter the DataFrame for nodes that should have connections to normal individuals
        nodes_to_connect = self.node_df[self.node_df['type'].isin(roles_with_normal_connections)]

        # Generate connections for each node
        connections = nodes_to_connect.apply(lambda row: self.generate_normal_connections(row, normal_people), axis=1)

        # Flatten the list of connections
        edge_list = [item for sublist in connections for item in sublist]

        return edge_list

    def generate_normal_connections(
            self,
            node_row: pd.DataFrame,
            normal_people: pd.DataFrame.index
            ) -> list:

        # Determine the number of normal connections (e.g., 1-3 for kingpins, 1-2 for dealers)
        if node_row['type'] == 'kingpin':
            num_connections = np.random.randint(1, self.kingpin_normal_connections)  # Kingpins have 1 to 3 normal connections
        else:  # Dealers
            num_connections = np.random.randint(1, self.dealer_normal_connections)  # Dealers have 1 to 10 normal connections

        # Select random normal individuals to connect with
        selected_normals = np.random.choice(normal_people, size=num_connections, replace=False)

        # Return a list of connections for the node
        return [(node_row.name, normal_index) for normal_index in selected_normals]

    def generate_edges(self):
        edge_list = []

        # Initial connections based on affiliations and roles
        # Ensure kingpin-dealer connectivity and dealer-user connections
        edge_list.extend(self.ensure_kingpin_dealer_connectivity())

        edge_list.extend(self.connect_dealers_to_users())

        # Within-group connections
        edge_list.extend(self.connect_within_group())

        # Random connections across the network
        edge_list.extend(self.connect_randomly())

        # Connect kingpins and dealers to normal individuals
        edge_list.extend(self.connect_to_normals())

        # Convert edge list to DataFrame
        self.edge_df = pd.DataFrame(edge_list, columns=['src', 'target'])

    def assign_personal_details(
            self,
            postal_code: str,
            state: str,
            max_num_whereabouts: int
            ) -> None:

        details_df = self.generate_details(
                num_records=len(self.node_df),
                postal_code=postal_code,
                state=state,
                num_whereabouts=max_num_whereabouts
                )
        self.node_df = pd.concat([self.node_df, details_df], axis=1)
        return self.expand_whereabouts_to_columns()

    def flatten_dict(self, d: dict) -> dict:
        items = []
        for key, value in d.items():
            if isinstance(value, dict):
                items.extend(self.flatten_dict(value).items())
            else:
                items.append((key, value))
        return dict(items)

    #PROFILE GENERATION
    def generate_details(
            self,
            num_records: int,
            postal_code: str,
            state: str,
            num_whereabouts: int
            ) -> pd.DataFrame:

        return pd.DataFrame([self.flatten_dict(profile.to_dict()) for profile in ProfileFactory.create_batch(num_records, postal_code=postal_code, state=state, num_whereabouts=num_whereabouts)])

    def expand_whereabouts_to_columns(self):
        max_whereabouts = self.max_num_whereabouts

        # Create a temporary DataFrame from the 'whereabouts' series
        whereabouts_df = self.node_df['whereabouts'].apply(pd.Series)

        # Iterate over the number of whereabouts
        for i in range(max_whereabouts):
            # Extract whereabouts details for each whereabouts
            whereabouts_details_df = whereabouts_df[i].apply(pd.Series)

            # Assign address, from_date, to_date, and other details to the node DataFrame
            self.node_df[f'whereabouts_{i+1}_address1'] = whereabouts_details_df['address1']
            self.node_df[f'whereabouts_{i+1}_address2'] = whereabouts_details_df['address2']
            self.node_df[f'whereabouts_{i+1}_city'] = whereabouts_details_df['city']
            self.node_df[f'whereabouts_{i+1}_state'] = whereabouts_details_df['state']
            self.node_df[f'whereabouts_{i+1}_postalCode'] = whereabouts_details_df['postalCode']
            self.node_df[f'whereabouts_{i+1}_coordinates'] = whereabouts_details_df['coordinates']
            # Flatten coordinates into lat and lng
            coordinates_df = whereabouts_details_df['coordinates'].apply(pd.Series)
            self.node_df[f'whereabouts_{i+1}_lat'] = coordinates_df['lat']
            self.node_df[f'whereabouts_{i+1}_lng'] = coordinates_df['lng']
            # Drop the coordinates column
            self.node_df.drop(f'whereabouts_{i+1}_coordinates', axis=1, inplace=True)

            self.node_df[f'whereabouts_{i+1}_from_date'] = whereabouts_details_df['from_date']
            self.node_df[f'whereabouts_{i+1}_to_date'] = whereabouts_details_df['to_date']


        # Drop the original 'whereabouts' column
        self.node_df.drop('whereabouts', axis=1, inplace=True)
        # Replace NaN values with None
        self.node_df = self.node_df.where(pd.notnull(self.node_df), None)

    @staticmethod
    def random_datetime(
        year: int,
        month: int,
        day: int,
        hour_start: int,
        hour_end: int
    ) -> datetime:

        start = datetime(year, month, day, hour_start)
        end = datetime(year, month, day, hour_end)
        return start + timedelta(
            seconds=np.random.randint(0, int((end - start).total_seconds()))
            )

    def assign_affiliations(self):
        # Step 1: Assign an affiliation to each kingpin
        kingpins = self.node_df[self.node_df['type'] == 'kingpin']

        shuffled_affiliations = np.random.choice(
            self.affiliations,
            size=len(self.affiliations),
            replace=False
            ).tolist()

        for i, index in enumerate(kingpins.index):
            if i < len(shuffled_affiliations):
                # Assign a unique affiliation to each kingpin
                self.node_df.at[index, 'affiliation'] = shuffled_affiliations[i]
            else:
                # If there are more kingpins than affiliations, assign random affiliations to the remaining kingpins
                self.node_df.at[index, 'affiliation'] = np.random.choice(self.affiliations)

        # Step 2: Calculate nearest kingpin for dealers and assign affiliations
        self.calculate_nearest_kingpin()
        # Ensure dealers inherit their kingpin's affiliation
        self.node_df.loc[self.node_df['type'] == 'dealer', 'affiliation'] = self.node_df.loc[self.node_df['type'] == 'dealer', 'nearest_kingpin_index'].map(lambda x: self.node_df.at[x, 'affiliation'])

        # Step 3: Assign 'None' to users and normal individuals
        self.node_df.loc[self.node_df['type'].isin(['user', 'normal']), 'affiliation'] = 'None'

    def generate_and_assign_criminal_records(self):
        unique_case_number = count(start=1000, step=1)  # Unique case number generator
        gang_related_cases = {}  # To track gang-related case numbers and crimes

        # Generate number of cases for each person
        self.node_df['num_cases'] = np.random.randint(
            0,
            self.max_cases_per_person + 1,
            size=len(self.node_df)
            )

        # Generate cases for each person
        self.node_df['cases'] = self.node_df.apply(
            lambda row: [
                self.generate_case(
                    row,
                    gang_related_cases,
                    unique_case_number
                )
                for _ in range(row['num_cases'])
                ],
            axis=1
            )

        # Drop the 'num_cases' column as it's no longer needed
        self.node_df.drop('num_cases', axis=1, inplace=True)
        return self.expand_cases_to_columns()

    def generate_case(
            self,
            person: pd.DataFrame,
            gang_related_cases: dict,
            unique_case_number: int
            ) -> dict:
        # Adjusted logic for determining shared or unique cases
        if person['affiliation'] != 'None' and gang_related_cases.get(person['affiliation']) and np.random.random() < self.shared_case_percentage:
            shared_case = np.random.choice(gang_related_cases[person['affiliation']])
            return shared_case
        else:
            case_num = next(unique_case_number)
            crimes_in_case = np.random.choice(
                self.crimes,
                np.random.randint(1, 4),
                replace=False
                ).tolist()

            new_case = {"case_number": case_num, "crimes": crimes_in_case}

            if person['affiliation'] != 'None':
                gang_related_cases.setdefault(
                    person['affiliation'],
                    []
                ).append(new_case)

            return new_case

    def expand_cases_to_columns(self):
        max_crimes_per_case = self.max_crimes_per_case  # Adjust based on your dataset

        # Create a temporary DataFrame from the 'cases' series
        cases_df = self.node_df['cases'].apply(pd.Series)

        # Iterate over the number of cases
        for i in range(max_crimes_per_case):
            # Extract case details for each case
            case_details_df = cases_df[i].apply(pd.Series)

            # Assign case number and crimes to the node DataFrame
            self.node_df[f'case_number_{i+1}'] = case_details_df['case_number'].astype('Int64')
            self.node_df[f'case_number_{i+1}'] = self.node_df[f'case_number_{i+1}'].astype('object')

            # Extract crimes for each case and assign to the node DataFrame
            crimes_df = case_details_df['crimes'].apply(pd.Series)
            for j in range(max_crimes_per_case):
                self.node_df[f'crime_{i+1}_{j+1}'] = crimes_df[j]

        # Drop the original 'cases' column
        self.node_df.drop('cases', axis=1, inplace=True)
        # Replace NaN values with None
        self.node_df = self.node_df.where(pd.notnull(self.node_df), None)

    #CALL LOG GENERATION
    def generate_phone_numbers(self):
        # Assuming self.node_df exists and has been populated
        self.teledict = self.node_df['phone'].to_dict()

    def generate_and_assign_call_logs(self, start_date, end_date):
        # Parse date strings
        start_date = datetime.strptime(start_date, '%Y-%m-%d') \
            if isinstance(start_date, str) else start_date

        end_date = datetime.strptime(end_date, '%Y-%m-%d') \
            if isinstance(end_date, str) else end_date

        # Ensure phone numbers are generated
        if not hasattr(self, 'teledict'):
            self.generate_phone_numbers()

        # Define a function to generate call logs for a given edge
        def generate_call_logs(edge: dict) -> list:
            # Check if the edge exists in self.edge_df
            if edge['src'] not in self.node_df.index or edge['target'] not in self.node_df.index:
                # If the edge doesn't exist, manually set the caller and callee types to 'kingpin'
                caller_type = 'kingpin'
                callee_type = 'kingpin'
            else:
                # If the edge does exist, get the caller and callee types from self.node_df
                caller_type = self.node_df.loc[edge['src'], 'type']
                callee_type = self.node_df.loc[edge['target'], 'type']

            # Determine the number of calls for this edge (e.g., 1-10)
            num_calls = np.random.randint(1, self.max_calls_per_edge)

            # Assign caller and callee phone numbers
            caller = self.teledict[edge['src']]
            callee = self.teledict[edge['target']]
            # Determine the number of calls for this edge (e.g., 1-10)

            # Check if the call is inter-gang (caller is a kingpin and callee is a dealer from diff gang)
            if caller_type == 'kingpin' and callee_type == 'dealer' and self.node_df.loc[edge['src'], 'affiliation'] != self.node_df.loc[edge['target'], 'affiliation']:
                call_type = 'inter-gang'
            # Check if the call is inter-gang (both nodes are kingpins from different gangs)
            elif caller_type == 'kingpin' and callee_type == 'kingpin' and self.node_df.loc[edge['src'], 'affiliation'] != self.node_df.loc[edge['target'], 'affiliation']:
                call_type = 'inter-gang'
            # Check if the call is intra-gang (caller is a kingpin and callee is a dealer from the same gang)
            elif caller_type == 'kingpin' and callee_type == 'dealer' and self.node_df.loc[edge['src'], 'affiliation'] == self.node_df.loc[edge['target'], 'affiliation']:
                call_type = 'intra-gang'
            #dealer to dealer intra-gang
            elif caller_type == 'dealer' and callee_type == 'dealer' and self.node_df.loc[edge['src'], 'affiliation'] == self.node_df.loc[edge['target'], 'affiliation']:
                call_type = 'intra-gang'
            # All other calls are non-affiliated
            else:
                call_type = 'non-affiliated'

            # Return a list of call logs for this edge
            return [{
                'src': edge['src'],
                'target': edge['target'],
                'caller': caller,
                'callee': callee,
                'call_time': self.random_datetime(
                    year=start_date.year + np.random.randint(0, (end_date - start_date).days // 365),
                    month=np.random.randint(1, 13),
                    day=np.random.randint(1, 29),
                    hour_start=0 if caller_type in ['user', 'normal'] else 8,
                    hour_end=23 if caller_type in ['user', 'normal'] else 22
                ).strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': np.random.randint(5, 61) if caller_type in ['user', 'normal'] else np.random.randint(1, 16),
                'call_type': call_type
            } for _ in range(num_calls)]

        # Generate call logs for each edge
        call_logs = self.edge_df.apply(generate_call_logs, axis=1).tolist()

        # Generate inter-gang calls between kingpins
        kingpins = self.node_df[self.node_df['type'] == 'kingpin']
        kingpin_calls = []

        for i in range(len(kingpins)):
            for j in range(i + 1, len(kingpins)):
                if kingpins.iloc[i]['affiliation'] != kingpins.iloc[j]['affiliation'] and np.random.random() < self.leader_to_leader_call_chance:  # 5% chance of a call
                    edge = {'src': kingpins.index[i], 'target': kingpins.index[j]}  # Use index here
                    kingpin_calls.append(generate_call_logs(edge))

        kg_calls = pd.DataFrame(kingpin_calls)
        call_logs_df = pd.DataFrame(call_logs)
        # Flatten the DataFrame
        flattened_df = pd.json_normalize(
            call_logs_df.apply(lambda x: x.tolist(), axis=1)
            .explode()
            .dropna()
            .tolist()
            )

        flattened_king_df = pd.json_normalize(
            kg_calls.apply(lambda x: x.tolist(), axis=1)
            .explode()
            .dropna()
            .tolist()
            )

        # Drop rows and columns that are entirely NaN
        flattened_df = flattened_df \
            .dropna(axis=0, how='all') \
            .dropna(axis=1, how='all')

        flattened_king_df = flattened_king_df \
            .dropna(axis=0, how='all') \
            .dropna(axis=1, how='all')

        # Assign the flattened DataFrame to self.edge_df
        self.edge_df = pd.concat([flattened_king_df, flattened_df])

    def to_graph(
            self,
            size_dict: dict = None,
            edge_influence: int = 7,
            icon_mapping: dict = None,
            color_mapping: dict = None
            ) -> graphistry.plotter.Plotter:

        ndf = self.node_df.copy()
        edf = self.edge_df.copy()

        edge_counts = edf.groupby(['src', 'target', 'call_type']) \
            .size() \
            .reset_index(name='weight')

        # Default size_dict if none is provided
        if size_dict is None:
            size_dict = {'kingpin': 200, 'dealer': 75, 'user': 50, 'normal': 25}

        ndf['size'] = ndf['type'].map(size_dict)

        # Default icon_mapping if none is provided
        if icon_mapping is None:
            icon_mapping = {
                'kingpin': 'user-o',
                'dealer': 'user-md',
                'user': 'users',
                'normal': 'universal-access',
            }

        # Default color_mapping if none is provided
        if color_mapping is None:
            color_mapping = {
                'non-affiliated': 'blue',
                'intra-gang': 'red',
                'inter-gang': 'orange'
                }

        g = (
            graphistry.nodes(ndf, 'node_id')
            .edges(edge_counts, 'src', 'target')
            .bind(point_title='type', point_size='size')
            .bind(edge_weight="weight", edge_color="call_type")
            .settings(url_params={'edgeInfluence': edge_influence})
            .encode_point_icon('type', categorical_mapping=icon_mapping)
            .encode_edge_color(
                'call_type',
                categorical_mapping=color_mapping,
                default_mapping='#CCC'
                )
        )

        return g

    def get_dealer_to_user_edges_and_nodes(
            self,
            affiliated_nodes: pd.DataFrame
            ) -> tuple:
        # Filter the node DataFrame to only include dealers
        affiliated_dealers = affiliated_nodes[affiliated_nodes['type'] == 'dealer']

        # Join the edges and nodes dataframes on the 'target' column
        edges_with_node_types = self.edge_df.merge(self.node_df[['node_id', 'type']], left_on='target', right_on='node_id', how='left')

        # Filter the joined dataframe to only include edges from dealers to users
        dealer_to_user_edges_df = edges_with_node_types[(edges_with_node_types['src'].isin(affiliated_dealers['node_id'])) & (edges_with_node_types['type'] == 'user')]

        # Create the dealer to user edges
        dealer_to_user_edges = dealer_to_user_edges_df[['src', 'target']].copy()
        dealer_to_user_edges['role'] = 'user'
        dealer_to_user_edges['affiliation'] = dealer_to_user_edges['src'].map(affiliated_dealers['affiliation'])

        # Get the user nodes
        user_nodes = self.node_df[self.node_df['node_id'].isin(dealer_to_user_edges['target'])]

        return dealer_to_user_edges, user_nodes

    def to_tree(self, affiliation: str) -> graphistry.plotter.Plotter:
        # Filter the node DataFrame by the specified affiliation
        affiliated_nodes = self.node_df[self.node_df['affiliation'] == affiliation].copy()
        affiliated_nodes.loc[:, "node_label"] = affiliated_nodes["first_name"] + " " + affiliated_nodes["last_name"]

        dealer_to_user_edges, user_nodes = self.get_dealer_to_user_edges_and_nodes(affiliated_nodes)

        user_nodes = pd.DataFrame(user_nodes)
        user_nodes.loc[:, "node_label"] = user_nodes["first_name"] + " " + user_nodes["last_name"]

        # Get the kingpin node
        kingpin_node = affiliated_nodes[affiliated_nodes['type'] == 'kingpin']['node_id'].values[0]

        # Add dealer nodes and edges to the dataframes based on the affiliations
        dealer_nodes = affiliated_nodes[affiliated_nodes['type'] == 'dealer']
        dealer_edges = pd.DataFrame({
            'src': kingpin_node,
            'target': dealer_nodes['node_id'],
            'role': dealer_nodes['type'],
            'affiliation': dealer_nodes['affiliation']
        })

        # Add dealer to user edges to the new_edges DataFrame
        new_edges = pd.concat([dealer_edges, dealer_to_user_edges])

        # Add user nodes to the new_nodes DataFrame
        new_nodes = pd.concat([affiliated_nodes, user_nodes])

        g = graphistry.bind(
            source='src',
            destination='target',
            node='node_id',
            point_title='node_label'
            ).edges(new_edges).nodes(new_nodes)
        g = g.encode_point_color('type', categorical_mapping={'kingpin': 'red', 'dealer': 'blue', 'user': 'green'}, default_mapping='gray')
        g = g.encode_point_icon('type', categorical_mapping={'kingpin': 'user-o', 'dealer': 'user-md', 'user': 'users'})
        g = g.settings(url_params={'play': 0, "edgeCurvature": 0.0})
        g = g.tree_layout(width=100, height=50)
        return g