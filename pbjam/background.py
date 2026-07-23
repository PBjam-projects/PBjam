"""
This module contains the model that is used to construct the background model
of the power density spectrum. 
"""

from pbjam import jar 

class bkgModel():
    """
    Evaluate the background model of a power density spectrum.
    
    Parameters
    ----------
    nu : array-like
        Frequency axis at which the model is evaluated.
    Nyquist : float
        Nyquist frequency in the same units as ``nu``.
    """
    def __init__(self, nu, Nyquist):
        """
        Evaluate the background model of a power density spectrum.
        
        Parameters
        ----------
        nu : array-like
            Frequency axis at which the model is evaluated.
        Nyquist : float
            Nyquist frequency in the same units as ``nu``.
        """
        self.nu = nu

        self.Nyquist = Nyquist

        self.eta = jar.attenuation(self.nu, self.Nyquist)**2
 
    def harvey(self, nu, a, b, c):
        """
        Evaluate a Harvey-like background component.
        
        Parameters
        ----------
        nu : array-like
            Frequency axis.
        a : float
            Power-scale parameter of the Harvey-like component.
        b : float
            Characteristic frequency.
        c : float
            Exponent controlling the high-frequency slope.
        
        Returns
        -------
        array-like
            Harvey-like component evaluated at ``nu``.
        """
         
        H = a / b * 1 / (1 + (nu / b)**c)

        return H
 
    def __call__(self, theta_u,):
        """
        Evaluate the complete background model.
        
        Parameters
        ----------
        theta_u : dict
            Background parameters. Required keys are ``H_power``, ``H1_nu``,
            ``H1_exp``, ``H2_nu``, ``H2_exp``, ``H3_power``, ``H3_nu``,
            ``H3_exp``, and ``shot``.
        
        Returns
        -------
        array-like
            Sum of the attenuated Harvey-like components and the shot-noise level.
        """

        H1 = self.harvey(self.nu, theta_u['H_power'], theta_u['H1_nu'], theta_u['H1_exp'],)

        H2 = self.harvey(self.nu, theta_u['H_power'], theta_u['H2_nu'], theta_u['H2_exp'],)

        H3 = self.harvey(self.nu, theta_u['H3_power'], theta_u['H3_nu'], theta_u['H3_exp'],)
            

        bkg = (H1 + H2 + H3) * self.eta + theta_u['shot']

        return bkg

