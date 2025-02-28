import numpy as np

def get_area_in_pixels(area_p:list[int], forward_length_m:int):
        '''A method to determine info to the vehicles'''
        ratio_p_div_m = area_p[1]/forward_length_m
        ID_to_area_p = {
            2:(np.array([1.75,4.55])*ratio_p_div_m).round(),
            5:(np.array([2.55,12])*ratio_p_div_m).round(),
            7:(np.array([2.4,16])*ratio_p_div_m).round()
        }
        return ID_to_area_p






        