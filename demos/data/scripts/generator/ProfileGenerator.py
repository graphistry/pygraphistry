import factory
from datetime import datetime, timedelta
import pandas as pd

import numpy as np
import random_address


class Profile:
    def __init__(
            self,
            firstname,
            lastname,
            phone_number,
            username, email,
            address,
            dob,
            whereabouts,
            num_whereabouts=None,
            postal_code=None,
            state=None,
            rand_num=None
            ):

        self.postal_code = postal_code
        self.state = state
        self.username = username
        self.email = email
        self.firstname = firstname
        self.lastname = lastname
        self.phone_number = phone_number
        self.address = address
        self.DOB = dob
        self.whereabouts = whereabouts
        self.rand_num = rand_num
        self.num_whereabouts = num_whereabouts

    def to_dict(self):
        return {"first_name": self.firstname,
                "last_name": self.lastname,
                "user_name": self.username,
                "DOB": self.DOB,
                "email": self.email,
                "phone": self.phone_number,
                "address": self.address,
                "whereabouts": self.whereabouts
                }

    def __str__(self):
        return str(self.__dict__)


#profile factory
class ProfileFactory(factory.Factory):
    class Meta:
        model = Profile
    # Optional parameters for address generation
    state = None
    postal_code = None
    num_whereabouts = None
    rand_num = factory.LazyFunction(lambda: str(np.random.randint(0, 999)))
    username = factory.LazyAttribute(lambda obj: f"{obj.firstname}.{obj.lastname}{obj.rand_num}".lower())
    email = factory.LazyAttribute(lambda obj: f"{obj.firstname}.{obj.lastname}@{str(obj.rand_num) + np.random.choice(pd.read_csv('domains.txt', header=None)[0].to_list())}".lower())
    dob = factory.LazyFunction(lambda: (datetime.today() - timedelta(days=np.random.randint(15 * 365, 85 * 365))).strftime('%m-%d-%Y'))
    firstname = factory.Faker('first_name')
    lastname = factory.Faker('last_name')
    phone_number = factory.Faker('basic_phone_number', locale="en_US")
    address = factory.LazyAttribute(lambda obj: ProfileFactory.generate_address(state=obj.state, postal_code=obj.postal_code, from_date=(datetime.today() - timedelta(days=np.random.randint(0, 365))).strftime('%m-%d-%Y'), to_date=datetime.today().strftime('%m-%d-%Y')))
    whereabouts = factory.LazyAttribute(lambda obj: [ProfileFactory.generate_address(state=obj.state, postal_code=obj.postal_code, from_date=(datetime.today() - timedelta(days=np.random.randint(365, 365 * 5))).strftime('%m-%d-%Y'), to_date=(datetime.today() - timedelta(days=np.random.randint(0, 365))).strftime('%m-%d-%Y')) for _ in range(obj.num_whereabouts)])
    
    @staticmethod
    def generate_address(state=None, postal_code=None, from_date=None, to_date=None) -> dict:
        """
        Function to generate an address.
        """
        if state and postal_code:
            raise ValueError("Cannot specify both state and postal code. Please choose one.")
        elif state:
            address = random_address.real_random_address_by_state(state)
        elif postal_code:
            address = random_address.real_random_address_by_postal_code(postal_code)
        else:
            address = random_address.real_random_address()

        # Add dates to address
        address['from_date'] = from_date
        address['to_date'] = to_date

        return address