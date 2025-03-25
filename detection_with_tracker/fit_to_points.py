from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from detection_with_tracker.test_open import open_image_in_vscode
from typing import Union
import os

from utils.timetester import timetester


# look into: https://chatgpt.com/share/67969aa3-53f8-8001-9db8-5b45fd5beef9

# decay rate++: https://chatgpt.com/share/6799f45c-55f4-8001-be53-3d693aec06dc

class Dataset:
    s1 = deque([2.503267471866802, 2.5125897489965365, 2.5187388304371248, 2.5176796535222885, 2.52109812590252, 2.5291431364729795, 2.5225594663215425, 2.5368900906060716, 2.542958822023699, 2.5518774666321664, 2.5568334914526427, 2.568710180311515, 2.591680816023493, 2.616033530703213, 2.6077451320163787, 2.6558420067946806, 2.626110523447077, 2.7878360409636866, 2.8212702698511634, 2.9269064324634915])
    t1 = deque([14.93465781211853, 14.983630418777466, 15.031429529190063, 15.07831335067749, 15.12585186958313, 15.173230409622192, 15.221158742904663, 15.26902985572815, 15.316233158111572, 15.363252878189087, 15.409212589263916, 15.493715524673462, 15.744398355484009, 15.793551921844482, 15.880655765533447, 16.023854970932007, 16.25991940498352, 16.948331356048584, 16.99699306488037, 17.3432354927063])

    s2 = deque([1.180001353308476, 1.1791379657747396, 1.1695980588985508, 1.1557191701141583, 1.1454137626164271, 1.1537147987536163, 1.1557459070499225, 1.149443335886563, 1.099832350331776, 1.0980822717447558, 1.1024486093742665, 1.0971470046239735, 1.0978695395840024, 1.059952642673923, 1.0527267771662963, 1.04654727373646, 1.045625671599644, 1.0457119635949335, 1.0598623257541577, 1.2115571720655856])
    t2 = deque([17.523476600646973, 17.568540573120117, 17.614941120147705, 17.664422035217285, 17.70947551727295, 17.75967526435852, 17.80578112602234, 17.850537538528442, 17.89599919319153, 17.942594528198242, 17.991590976715088, 18.038654804229736, 18.085050106048584, 18.132141590118408, 18.383930206298828, 18.549413919448853, 18.875269412994385, 19.003410816192627, 19.08923602104187, 19.739616870880127])

    s3 = deque([3.5730248945506347, 3.5960351175304446, 3.5778543198751978, 3.5436986147252343, 3.4779369470541512, 3.4858238674221473, 3.392082476131694, 3.36418562598596, 3.3000681417445747, 3.2988089081427945, 3.2376479587568077, 3.245524313017992, 3.211885378943419, 3.208476696244699, 3.1869594419884404, 3.1992323750444203, 3.1225454170059757, 3.1184314640441513, 3.1510518545785615, 3.1184362506916883])
    t3 = deque([10.231421709060669, 10.278669595718384, 10.37704586982727, 10.424573183059692, 10.471928358078003, 10.519058227539062, 10.569148778915405, 10.615509033203125, 10.665026664733887, 10.715932369232178, 10.857682228088379, 10.906960487365723, 10.95359492301941, 11.187206268310547, 11.341896295547485, 11.388872385025024, 11.531938552856445, 11.578141689300537, 11.626359462738037, 11.674144268035889])

    def give_data(index:int):
        s = [Dataset.s1,Dataset.s2,Dataset.s3][index]
        t = [Dataset.t1,Dataset.t2,Dataset.t3][index]
        return s,t

class UpdatedData:
    data = {
        1:{
            'd':deque([3.4034437902633394, 3.3229657759383335, 3.339162906590546, 3.2115489019420047, 3.138862194590619, 3.11057370080868, 3.022579998160743, 2.9803361671988284, 2.9072821004504132, 2.8530010709553095, 2.8661374055601514, 2.7646215328110815, 2.6232178647528537, 2.5439719509141, 2.448369130075666, 2.4680238554930933, 2.393236415639552, 2.3036609349860293, 2.225563724451451, 2.413466887111379]),
            'dx':deque([1.2932957464093058, 1.2657476072621678, 1.2797610447895202, 1.238120487609066, 1.214386751371361, 1.212618577734642, 1.1803501906087197, 1.1717530776763305, 1.1487080374711334, 1.1307812965134656, 1.1398395871648594, 1.12174461473418, 1.0908057759605592, 1.0613205487419495, 1.0393592046441034, 1.063313040195568, 1.0366161153142983, 1.0009978244747244, 0.975537303301667, 1.080061684764094]),

            'dy':deque([3.1481448101702187, 3.0724557836960886, 3.0841888050165576, 2.9632927644307414, 2.894429251981114, 2.8644763802648856, 2.7825821232825017, 2.7403281545226386, 2.6707225719360266, 2.619341285576866, 2.6297356032656434, 2.5268203812296512, 2.3856686117500905, 2.312010376243104, 2.2168093830613964, 2.2272195962304773, 2.157083116297085, 2.074814945670475, 2.0003651820258033, 2.1583070152987123]),
            't':deque([0.2986564636230469, 0.34808897972106934, 0.39760303497314453, 0.44808459281921387, 0.4978005886077881, 0.5476531982421875, 0.5971758365631104, 0.6476304531097412, 0.6997992992401123, 0.7505092620849609, 0.8001377582550049, 0.9195058345794678, 1.0869636535644531, 1.13749098777771, 1.288707971572876, 1.3891522884368896, 1.4396085739135742, 1.4900901317596436, 1.5402050018310547, 1.6912600994110107])
        }
    }
class PolyFitting:        
    def __init__(self, degree:int = 1, weight_function_info: dict[float, str] = {'min_weight':0.1, 'max_weight':1, 'scale_factor':1, 'decay_rate':1,'mode':'linear'}, viewing=False, saving=False):
        '''Check for time untill crash. 

        args:
            degree: int = the degree of the polynom which is fitted to
            weight_function_info: dict = A dictionary of the different parameters for the weight function
                max_weight / min_weight: float = You can control the range of weights by adjusting min_weight and max_weight.
                scale_factor: float = You can scale the weights using the scale_factor.
                mode: str = You can switch between 'linear' and 'exponential' weighting with the mode parameter.
            viewing: bool = If one should view the results for each detection. 
            saving: bool = if the view results should get saved

        '''
        self.weight_function_info:dict = weight_function_info

        self.degree = degree 

        self.viewing = viewing
        self.saving = saving
       
    def weight_function(self,t:deque):
        # Defaults for the extra parameter
        min_weight = self.weight_function_info.get('min_weight', 0.1)
        max_weight = self.weight_function_info.get('max_weight', 1.0)
        scale_factor = self.weight_function_info.get('scale_factor', 1.0)
        mode = self.weight_function_info.get('mode', 'linear')  # 'linear' or 'exponential'

        # Ensure time is not empty
        if len(t) == 0:
            return np.array([])

        # Linear weighting (no changes needed here)
        if mode == 'linear':
            return np.linspace(min_weight, max_weight, len(t)) * scale_factor

        # Exponential weighting (with time-based decay)
        elif mode == 'exponential':
            # Calculate the exponential decay based on time differences (relative to the last time point)
            time_diff = np.array(t) - t[-1]  # Difference from the last time point
            decay_rate = self.weight_function_info.get('decay_rate', 0.1)  # Decay rate parameter

            # Apply exponential decay based on time difference
            decay_weights = np.exp(-decay_rate * time_diff)  # Exponential decay

            # Normalize weights to the range [min_weight, max_weight]
            decay_weights = (decay_weights - np.min(decay_weights)) / (np.max(decay_weights) - np.min(decay_weights))
            decay_weights = min_weight + (max_weight - min_weight) * decay_weights  # Scale to desired range

            # Apply scale factor to final weights
            return decay_weights * scale_factor

        else:
            raise ValueError("Invalid mode. Choose between 'linear' and 'exponential'")

    def get_regression_model(self, d:deque, t:deque) -> np.ndarray[np.float64]:
        '''Fits the plot to the points.
        
        Returns the coefficients for the plot, which is a polynomial'''
        # Here we use w=self.weight_function(), since it has already been specified, we just run it
        weights = self.weight_function(t)
        coeff = np.polyfit(t, d, self.degree, w=weights)
        return coeff

    def get_values_from_model(self, coeffs, t:deque):
        '''Returns point from plot and value'''
        d:np.ndarray[float] = np.polyval(coeffs, t)
        return d


def plot_regression(index: int, future_time: float = 3, save_path: str = "regression_plot_white.png"):
    # Get data
    s, t = Dataset.give_data(index)
    
    # Convert deque to numpy arrays
    s = np.array(s)
    t = np.array(t)
    
    # Fit linear regression
    poly_fit = PolyFitting(degree=1)
    coeffs = poly_fit.get_regression_model(s, t)
    
    # Generate regression line
    t_fit = np.linspace(min(t), max(t) + future_time, 100)
    s_fit = np.polyval(coeffs, t_fit)
    
    # Predict future value
    future_s = np.polyval(coeffs, max(t) + future_time)
    
    # Set dark mode
    
    # Plot data
    plt.figure(figsize=(8, 6))
    plt.scatter(t, s, label='Original Data', color='blue', zorder = 3)
    plt.scatter([max(t) + future_time], [future_s], color='lime', label=f'Prediction (+{future_time}s)', zorder=4)

    plt.plot(t_fit, s_fit, label='Linear Regression', color='red', zorder = 2)
    
    # Labels and legend
    plt.xlabel('Time (t)', color='black')
    plt.ylabel('Distance (d)', color='black')
    plt.title(f'Linear Regression and Predictions', color='white')
    plt.legend()
    plt.grid(color='black')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    'Example of how it might look'
    plot_regression(0)

