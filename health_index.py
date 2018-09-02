import numpy as np

class TargetHIMatcher():

    def __init__(self):
        self.__lookup = {}

    def tell(self, indices, errors):

        for k in range(np.shape(indices)[0]):
            instance_index, time_index, error = indices[k,0], indices[k,1], errors[k]

            if not instance_index in self.__lookup:
                self.__lookup[instance_index] = {}

            if not time_index in self.__lookup[instance_index]:
                self.__lookup[instance_index][time_index] = []

            self.__lookup[instance_index][time_index].append(error)

    def finalize(self):

        target_hi = []

        for instance_index, value_dict in self.__lookup.items():
            max_index = max(self.__lookup[instance_index].keys())
            health_values = np.zeros([max_index])

            for time_index, values in value_dict.items():
                health_values[time_index - 1] = np.mean(values)

            target_hi.append(TargetHI(health_values= health_values, instance_num= instance_index))

        return target_hi



class TargetHI():

    def __init__(self, health_values, instance_num):
        self.__health_values = health_values
        self.instance_num = instance_num

        # normalize the reconstruction error
        self.__health_values = (np.max(self.__health_values) - self.__health_values) / (np.max(self.__health_values) - np.min(self.__health_values))

        # square
        self.__health_values = np.square(self.__health_values)

    def get_lifetime(self):
        return np.shape(self.__health_values)[0]

    def similartiy(self,other, t, tau):
        assert( self.get_lifetime() >= other.get_lifetime() + t )
        distance = np.sum(np.square(other.__health_values - self.__health_values[t:t+other.get_lifetime()]))
        return np.exp(- distance / other.get_lifetime() / tau)

    def rul_estimate(self, other, t):
        return self.get_lifetime() - other.get_lifetime() - t

    def health(self):
        return self.__health_values
