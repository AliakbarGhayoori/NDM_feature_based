import numpy as np
from datetime import datetime
import time
import calendar


def convert_date_to_seconds(date_str):
    date = date_str.split()
    month = list(calendar.month_abbr).index(date[1])
    day = int(date[2])
    year = int(date[5])
    time_arr = date[3].split(':')
    t = datetime(year, month , day, int(time_arr[0]), int(time_arr[1]) , int(time_arr[2]))
    return int(time.mktime(t.timetuple()))




class User(object):
    def __init__(self, user):
        self.user = user
        super(User, self).__init__()
    
    @property
    def vector(self):
        return np.array([self.length_of_user_description,
                         self.length_of_username,
                         self.friends_count,
                         self.statuses_count,
                         self.registration_age,
                         self.is_verified,
                         self.is_geo_enabled,
                         self.gender])
    
    @property
    def length_of_user_description(self):
        # TODO: make it correctly
        if self.user['user_description']:
            return len(self.user['user_description'])
        else:
            return 0
    
    @property
    def length_of_username(self):
        # TODO: make it correctly
        return len(self.user['screen_name'])
    
    @property
    def followers_count(self):
        return self.user['followers_count']
    
    @property
    def friends_count(self):
        return self.user['friends_count']
    
    @property
    def statuses_count(self):
        return self.user['statuses_count']
    
    @property
    def registration_age(self):
        now = time.time()
        days = (now - self.user['user_created_at'])
        
        return int(days)

    
    @property
    def is_verified(self):
        # verified: 1
        # not verified: 0
        
        return 1 if self.user['verified'] else 0
    
    @property
    def is_geo_enabled(self):
        # enabled: 1
        # not enabled: 0
        return 1 if self.user['user_geo_enabled'] else 0
    
    @property
    def gender(self):
        return 1 if self.user['gender']=='m' else 0
        
    
    @property
    def time_of_tweet(self):
        return self.user['t']


    @property
    def id(self):
        return self.user['uid']

    @property
    def text(self):
        return self.user['text']



class TUser(object):
    def __init__(self, user):
        self.user = user
        super(TUser, self).__init__()
    
    @property
    def vector(self):
        return np.array([self.length_of_user_description,
                         self.length_of_username,
                         self.followers_count,
                         self.friends_count,
                         self.statuses_count,
                         self.registration_age,
                         self.is_verified,
                         self.is_geo_enabled])
    
    @property
    def length_of_user_description(self):
        # TODO: make it correctly
        if self.user['description']:
            return len(self.user['description'])
        else:
            return 0
    
    @property
    def length_of_username(self):
        # TODO: make it correctly
        return len(self.user['name'])
    
    @property
    def followers_count(self):
        return self.user['followers_count']
    
    @property
    def friends_count(self):
        return self.user['friends_count']
    
    @property
    def statuses_count(self):
        return self.user['statuses_count']
    
    @property
    def registration_age(self):
        now = time.time()
        days = now -  convert_date_to_seconds( self.user['created_at'])
        
        return int(days)
        return  0
    
    @property
    def is_verified(self):
        # verified: 1
        # not verified: 0
        
        return 1 if self.user['verified'] else 0
    
    @property
    def is_geo_enabled(self):
        # enabled: 1
        # not enabled: 0
        return 1 if self.user['geo_enabled'] else 0
    
    @property
    def time_of_tweet(self):
        return convert_date_to_seconds(self.user['created_at'])
    
    @property
    def gender(self):
        return 1 if self.user['gender']=='m' else 0

    @property
    def id(self):
        return self.user['id']



__all__ = ['User' , 'TUser']
