import numpy as np

from bdm_voxel_builder.helpers.numpy import create_random_array, create_zero_array, get_mask_zone_xxyyzz
from bdm_voxel_builder.helpers.math import remap

class DataLayer:
    def __init__(self, name = '', voxel_size = 20, rgb = [1,1,1], 
                diffusion_ratio = 0.12, diffusion_random_factor = 0,
                decay_ratio = 0, decay_random_factor = 0, decay_linear_value = 0, emission_factor = 0.1,
                gradient_resolution = 0, color_array = None, flip_colors = False):
        
        # self._array =  # value array
        self._color_array = None # colored array 4D
        self._axis_order = 'zyx'
        self._name = name
        self._n = voxel_size
        self._array = np.zeros(self._n ** 3).reshape([self._n, self._n, self._n])
        self._rgb = rgb
        self._diffusion_ratio = diffusion_ratio
        self._diffusion_random_factor = diffusion_random_factor
        self._decay_ratio = decay_ratio
        self._decay_random_factor = decay_random_factor
        self._decay_linear_value = decay_linear_value
        self._gradient_resolution = gradient_resolution
        self._voxel_crop_range = [0,1] # too much
        self._iter_count = 0
        self._color_array = color_array
        self._emission_factor = emission_factor
        self._emmision_array = None
        self._flip_colors = flip_colors
        self._gravity_ratio = 0
    
    def __str__(self):
        properties = []
        count = 0
        for name in dir(self):
            if isinstance(getattr(self.__class__, name, None), property):
                # Conditionally exclude properties from the string
                if name == 'array' or name == "color_array":  # Example condition: exclude 'voxel_size' property
                    pass
                else:
                    value = getattr(self, name)
                    properties.append(f"{name}={value}")
                    # if (count % 2) == 1:
                    properties.append('\n')
                    count += 1
        return f"{self.__class__.__name__}({', '.join(properties)})"
    


    def get_params(self):
        text = str(self.__str__)
        return text
    
    # Property getters
    @property
    def array(self):
        return self._array
    
    @property
    def color_array(self):
        self.calculate_color_array_remap()
        return self._color_array
    
    @property
    def name(self):
        return self._name

    @property
    def voxel_size(self):
        return self._n

    @property
    def rgb(self):
        return self._rgb

    @property
    def diffusion_ratio(self):
        return self._diffusion_ratio

    @property
    def diffusion_random_factor(self):
        return self._diffusion_random_factor

    @property
    def decay_ratio(self):
        self._decay_ratio = abs(self._decay_ratio * -1)
        return self._decay_ratio

    @property
    def decay_random_factor(self):
        return self._decay_random_factor

    @property
    def decay_linear_value(self):
        self._decay_linear_value = abs(self._decay_linear_value * -1)
        return self._decay_linear_value

    @property
    def axis_order(self):
        return self._axis_order

    @property
    def gradient_resolution(self):
        return self._gradient_resolution

    @property
    def iter_count(self):
        return self._iter_count
    
    @property
    def emission_factor(self):
        return self._emission_factor

    @property
    def voxel_crop_range(self):
        return self._voxel_crop_range

    @property
    def flip_color(self):
        return self._flip_colors

    # Property setters

    @emission_factor.setter
    def emission_factor(self, value):
        self._emission_factor = value

    @flip_color.setter
    def flip_color(self, value):
        self._flip_colors = value
    
    @iter_count.setter
    def iter_count(self, value):
        if not isinstance(value, float):
            raise ValueError("value must be a float")
        self._iter_count = value

    @array.setter
    def array(self, a):
        """Setter method for size property"""
        if not isinstance(a, np.ndarray):
            raise ValueError("must be a np.ndarray instance")
        if np.ndim(a) != 3:
            raise ValueError("Array should be 3D, in shape [n,n,n]")
        self._array = a

    @color_array.setter
    def color_array(self, a):
        """Setter method for size property"""
        if not isinstance(a, np.ndarray):
            raise ValueError("Size must be a np.ndarray instance")
        if np.ndim(a) != 3:
            raise ValueError("Array should be 4D, in shape [n, n, n, 3]")
        self._array = a
    
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("Name must be a string")
        self._name = value

    @voxel_size.setter
    def voxel_size(self, value):
        if not isinstance(value, (int)):
            raise ValueError("Voxel size must be an integrer")
        self._n = value

    @rgb.setter
    def rgb(self, a):
        """Setter method for size property"""
        if not isinstance(a, (list, tuple, np.ndarray)):
            raise ValueError("rgb must be a list of three floats")
        self._rgb = a

    @diffusion_ratio.setter
    def diffusion_ratio(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Diffusion ratio must be a number")
        self._diffusion_ratio = value

    @property
    def gravity_dir(self):
        return self._gravity_dir
    
    @gravity_dir.setter
    def gravity_dir(self, v):
        """direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
        if not isinstance(v, (int)) or v > 5 or v < 0:
            raise ValueError("gravity ratio must be an integrer between 0 and 5")
        self._gravity_dir = v

    @property
    def gravity_ratio(self):
        return self._gravity_ratio
    
    @gravity_ratio.setter
    def gravity_ratio(self, v):
        if not isinstance(v, (int, float)):
            raise ValueError("gravity ratio must be a number")
        self._gravity_ratio = v
    
  

    @diffusion_random_factor.setter
    def diffusion_random_factor(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Diffusion random factor must be a number")
        if not 0 <= value:
            raise ValueError('Diffusion random factor must be non-negative')
        self._diffusion_random_factor = value

    @decay_ratio.setter
    def decay_ratio(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Decay ratio must be a number")
        if not 0 <= value:
            raise ValueError('Decay ratio must be non-negative')
        self._decay_ratio = value

    @decay_linear_value.setter
    def decay_linear_value(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Decay linear ratio must be a number")
        self._decay_linear_value = abs(value * -1)

    @decay_random_factor.setter
    def decay_random_factor(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Decay random factor must be a number")
        self._decay_random_factor = value

    @axis_order.setter
    def axis_order(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Axis order must be a list or tuple")
        self._axis_order = value

    @gradient_resolution.setter
    def gradient_resolution(self, value):
        if not isinstance(value, (int)) or value < 0 :
            raise ValueError("Gradient resolution must be a nonnegative integrer; formula: y = int(1 * x) / x")
        self._gradient_resolution = value
    
    @voxel_crop_range.setter
    def voxel_crop_range(self,value):
        if not(isinstance(list, tuple)):
            raise ValueError('range must be a list or tuple of floats')
        self._voxel_crop_range = value

    def empty_array(self):
        self._array = np.zeros(self._n ** 3).reshape([self._n, self._n, self._n])
    
    def random(self, add = 0, strech = 1, crop = False, start = 0, end = 1):
        self._array = (np.random.random(self._n ** 3).reshape(self._n,self._n,self._n) + add) * strech
        if crop:
            self._array = self.crop_array(self._array, start ,end)

    def crop_array(self, array, start = 0, end = 1):
        array = np.minimum(array, end)
        array = np.maximum(array, start)
        return array
    
    def conditional_fill(self, condition = '<', value = 0.5, override_self = False):
        """returns new voxel_array with 0,1 values based on condition"""
        if condition == '<':
            mask_inv = self.array < value
        elif condition == '>':
            mask_inv = self.array > value
        elif condition == '<=':
            mask_inv = self.array <= value
        elif condition == '>=':
            mask_inv = self.array >=  value
        a = create_zero_array(self._n)
        a[mask_inv] = 0
        if override_self:
            self.array = a
        return a

    def set_layer_value_at_index(self, index = [0,0,0], value = 1):
        index2 = np.mod(index, self.voxel_size)
        i,j,k = index2
        self.array[i][j][k] = value
        return self.array

    def get_value_at_index(self, index = [0,0,0]):
        i,j,k = index
        v = self.array[i][j][k]
        return v

    def get_nonzero_point_list(self, array):
        """returns indicies of nonzero values
        if list_of_points:
            shape = [n,3]
        else:
            shape = [3,n]"""
        non_zero_array = np.nonzero(array)
        return np.transpose(non_zero_array)
    
    def get_nonzero_index_coordinates(self, array):
        """returns indicies of nonzero values
        list of coordinates
            shape = [3,n]"""
        non_zero_array = np.nonzero(array)
        return non_zero_array
    
    def grade(self):
        if self.gradient_resolution == 0:
            pass
        else:
            self._array = np.int64(self.array * self._gradient_resolution) / self._gradient_resolution
    
    def diffuse(self, limit_by_Hirsh = True, reintroduce_on_the_other_end = False):
        """infinitive borders
        every value of the voxel cube diffuses with its face nb
        standard finite volume approach (Hirsch, 1988). 
        in not limit_by_Hirsch: ph volume can grow
        diffusion change of voxel_x between voxel_x and y:
        delta_x = -a(x-y) 
        where 0 <= a <= 1/6 
        """
        if limit_by_Hirsh:
            self._diffusion_ratio= max(0, self._diffusion_ratio)
            self._diffusion_ratio= min(1/6, self._diffusion_ratio)
        
        shifts = [-1, 1]
        axes = [0,0,1,1,2,2] 
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = create_zero_array(self._n)
        for i in range(6):
            # y: shift neighbor
            y = np.copy(self._array)
            y = np.roll(y, shifts[i % 2], axis = axes[i])
            if not reintroduce_on_the_other_end:
                e = self._n - 1
                # removing the values from the other end after rolling
                if i == 0:
                    y[:][:][e] = 0
                elif i == 1:
                    y[:][:][0] = 0
                elif 2 <= i <= 3:
                    m = y.transpose((1,0,2))
                    if i == 2:
                        m[:][:][e] = 0
                    elif i == 3:
                        m[:][:][0] = 0
                    y = m.transpose((1,0,2))
                elif 4 <= i <= 5:
                    m = y.transpose((2,0,1))
                    if i == 4:
                        m[:][:][e] = 0
                    elif i == 5:
                        m[:][:][0] = 0
                    y = m.transpose((1,2,0))
            # calculate diffusion value
            if self._diffusion_random_factor == 0:
                diff_ratio = self.diffusion_ratio
            else:
                diff_ratio = self.diffusion_ratio * (1 - create_random_array(self._n) * self.diffusion_random_factor)
            # summ up the diffusions per faces
            total_diffusions += diff_ratio * (self._array - y) / 2
        self._array -= total_diffusions
        return self._array
    

    def gravity_shift(self, reintroduce_on_the_other_end = False):
        """direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up
        infinitive borders
        every value of the voxel cube diffuses with its face nb
        standard finite volume approach (Hirsch, 1988). 
        diffusion change of voxel_x between voxel_x and y:
        delta_x = -a(x-y) 
        where 0 <= a <= 1/6 
        """
        
        shifts = [-1, 1]
        axes = [0,0,1,1,2,2] 
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = create_zero_array(self._n)
        if self.gravity_ratio != 0:
            for i in [self.gravity_dir]:
                # y: shift neighbor
                y = np.copy(self._array)
                y = np.roll(y, shifts[i % 2], axis = axes[i])
                if not reintroduce_on_the_other_end:
                    # TODO replace to padded array method
                    e = self._n - 1
                    # removing the values from the other end after rolling
                    if i == 0:
                        y[:][:][e] = 0
                    elif i == 1:
                        y[:][:][0] = 0
                    elif 2 <= i <= 3:
                        m = y.transpose((1,0,2))
                        if i == 2:
                            m[:][:][e] = 0
                        elif i == 3:
                            m[:][:][0] = 0
                        y = m.transpose((1,0,2))
                    elif 4 <= i <= 5:
                        m = y.transpose((2,0,1))
                        if i == 4:
                            m[:][:][e] = 0
                        elif i == 5:
                            m[:][:][0] = 0
                        y = m.transpose((1,2,0))
                total_diffusions += self.gravity_ratio * (self._array - y) / 2
            self._array -= total_diffusions
        else:
            pass
        return self._array
    

    def emission_self(self, proportional = True):
        """updates array values based on self array values
        by an emission factor ( multiply / linear )"""

        if proportional: #proportional
            self.array += self.array * self.emission_factor
        else: # absolut
            self.array = np.where(self.array != 0, self.array + self.emission_factor, self.array)
    

    def emission_intake(self, external_emission_array, factor, proportional = True):
        """updates array values based on a given array
        and an emission factor ( multiply / linear )"""

        if proportional: #proportional
            # self.array += external_emission_array * self.emission_factor
            self.array = np.where(external_emission_array != 0, self.array + external_emission_array * factor, self.array)
        else: # absolut
            self.array = np.where(external_emission_array != 0, self.array + factor, self.array)
    
    def block_layers(self, other_layers = []):
        """acts as a solid obstacle, stopping diffusion of other layers
        input list of layers"""
        for i in range(len(other_layers)):
            layer = other_layers[i]
            layer.array = np.where(self.array == 1, 0 * layer.array, 1 * layer.array)
        pass

    def decay(self):
        if self._decay_random_factor == 0:
            self.array -= self.array *  self._decay_ratio
        else:
            randomized_decay = self._decay_ratio * (1 - create_random_array(self._n) * self._decay_random_factor)
            randomized_decay = abs(randomized_decay) * -1
            self.array += self.array * randomized_decay

    def decay_linear(self):
        s,e = self.voxel_crop_range
        self._array = self.crop_array(self._array - self.decay_linear_value, s,e)

    def calculate_color_array_remap(self):
        r,g,b = self.rgb
        colors = np.copy(self.array)
        min_ = np.min(colors) 
        max_ = np.max(colors)
        colors = remap(colors, output_domain = [0,1], input_domain = [min_, max_])
        if self._flip_colors:
            colors = 1 - colors

        reds = np.reshape(colors * (r), [self._n, self._n, self._n, 1])
        greens = np.reshape(colors * (g), [self._n, self._n, self._n, 1])
        blues = np.reshape(colors * (b), [self._n, self._n, self._n, 1])
        colors = np.concatenate((reds, greens, blues), axis = 3)
        self._color_array = colors
        return self._color_array

    def iterate(self, diffusion_limit_by_Hirsh=False, reintroduce_on_the_other_end=False ):
        self.iter_count += 1
        # emission update
        self.emmission_in()
        # decay
        self.decay()
        # diffuse
        self.diffuse(diffusion_limit_by_Hirsh, reintroduce_on_the_other_end)
        #emission_out
        self.emmission_out_update()
    
    def get_merged_array_with(self, other_layer):
                a1 = self.array
                a2 = other_layer.array
                return a1 + a2


    def add_values_in_zone_xxyyzz(self, zone_xxyyzz, value = 1, add_values = False):
        """add or replace values within zone (including both end)
        add_values == True: add values in self.array
        add_values == False: replace values in self.array *default
        input: 
            zone_xxyyzz = [x_start, x_end, y_start, y_end, z_start, z_end]
            """
        # np.zeros_like(self.array)
        n = self.voxel_size
        
        if add_values:
            zone = get_mask_zone_xxyyzz(self.voxel_size, zone_xxyyzz, return_bool = False)
            zone *= value
            self.array += zone
        else:
            mask = get_mask_zone_xxyyzz(self.voxel_size, zone_xxyyzz, return_bool=True)
            self.array[mask] = value

