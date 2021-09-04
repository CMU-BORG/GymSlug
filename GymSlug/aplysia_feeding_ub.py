import numpy as np
#%matplotlib qt
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
import gym
import random
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

import numpy as np
#%matplotlib qt
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

class AplysiaFeedingUB:
    #Timing variables
    TimeStep = 0.05            #time step in seconds
    StartingTime = 0           #simulation start time (in seconds)
    EndTime = 60               #simulation  time (in seconds)

    #Maximum muscle forces
    max_I4 = 1.75              #Maximum pressure grasper can exert on food
    max_I3ant = 0.6            #Maximum I3 anterior force
    max_I3 = 1                 #Maximum I3 force
    max_I2 = 1.5               #Maximum I2 force
    max_hinge = 0.2            #Maximum hinge force

    #Muscle time constants
    tau_I4 = 1.0/np.sqrt(2)              #time constant (in seconds) for I4 activation
    tau_I3anterior = 2.0/np.sqrt(2)      #time constant (in seconds) for I3anterior activation
    tau_I2_ingestion = 0.5*1/np.sqrt(2)  #time constant (in seconds) for I2 activation during ingestion
    tau_I2_egestion = 1.4*1/np.sqrt(2)   #time constant (in seconds) for I2 activation during egestion
    tau_I3 = 1.0/np.sqrt(2)              #time constant (in seconds) for I3 activation
    tau_hinge  = 1.0/np.sqrt(2)          #time constant (in seconds) for hinge activation

    #body time constants
    c_g = 1.0                  #time constant (in seconds) for grapser motion
    c_h = 1.0                  #time constant (in seconds) for body motion

    #Spring constants
    K_sp_h = 2.0       #spring constant representing neck and body between head and ground
    K_sp_g = 0.1       #spring constant representing attachment between buccal mass and head

    #Reference points for springs
    x_h_ref = 0.0      #head spring reference position
    x_gh_ref = 0.4     #grasper spring reference position

    #Friction coefficients
    mu_s_g = 0.4               #mu_s coefficient of static friction at grasper
    mu_k_g = 0.3               #mu_k coefficient of kinetic friction at grasper
    mu_s_h = 0.3               #mu_s coefficient of static friction at jaws
    mu_k_h = 0.3               #mu_k coefficient of kinetic friction at jaws

    #Sensory feedback thresholds (theshold_neuron name_behavior_type)
    thresh_B64_bite_protract = 0.89
    thresh_B64_swallow_protract = 0.4
    thresh_B64_reject_protract = 0.5

    thresh_B4B5_protract = 0.7

    thresh_B31_bite_off = 0.55
    thresh_B31_swallow_off = 0.4
    thresh_B31_reject_off = 0.6
    thresh_B31_bite_on = 0.9
    thresh_B31_swallow_on = 0.75
    thresh_B31_reject_on = 0.89

    thresh_B7_bite_protract = 0.9
    thresh_B7_reject_protract = 0.7

    thresh_B6B9B3_bite_pressure = 0.2
    thresh_B6B9B3_swallow_pressure = 0.25
    thresh_B6B9B3_reject_pressure = 0.75

    thresh_B38_retract = 0.4

    #neural state variables
    MCC=[]
    CBI2=[]
    CBI3=[]
    CBI4=[]
    B64=[]
    B4B5=[]
    B40B30=[]
    B31B32=[]
    B6B9B3=[]
    B8=[]
    B7=[]
    B38=[]
    B20=[]

    #neural timing variables
    refractory_CBI3 = 5000                 #refractory period (in milliseconds) of CBI3 post strong B4B5 excitation
    postActivityExcitation_B40B30 = 3000   #time (in milliseconds) post B40B30 activity that slow excitation lasts
    CBI3_stimON = 0
    CBI3_stimOFF = 0
    CBI3_refractory = 0
    B40B30_offTime = 0
                    

    #muscle state variables
    P_I4=[]
    A_I4=[]
    P_I3_anterior=[]
    A_I3_anterior=[]
    T_I3=[]
    A_I3=[]
    T_I2=[]
    A_I2=[]
    T_hinge=[]
    A_hinge=[]

    #body state variables
    x_h=[]
    x_g=[]
    grasper_friction_state =[]     #0 = kinetic friction, 1 = static friction
    jaw_friction_state =[]       #0 = kinetic friction, 1 = static friction
    theta_g =[]

    #environment variables
    seaweed_strength = 10
    fixation_type = 1          #default initialization is seaweed fixed to the force transducer (use for swallowing)
    force_on_object=[]

    #sensory state vectors
    sens_chemical_lips=[]
    sens_mechanical_lips=[]
    sens_mechanical_grasper=[]

    #switches
    use_hypothesized_connections = 0 #1 = yes, 0 = no

    #stimulation electrodes
    stim_B4B5=[] #0 = off, 1 = on
    stim_CBI2=[] #0 = off, 1 = on
    
    #variable to specify the desired biomechanical model
    biomechanicsModel = 1 #1 = Biomechanics_001 (1D), 2 = Biomechanics_002 (1.5D)
    neuralModel = 1
    muscleModel = 1
    
    ##properties for visualization
    
    #define the location of the ground plane
    x_ground = np.array([[0],[0]])
    len_ground_line = 5

    #define the location of the force transducer
    x_transducer = x_ground + np.array([[8],[0]])
    len_transducer_line = 5

    #define location and shape of head
    x_H = x_ground + np.array([[0],[0]])
    x_H_width = 1
    x_H_height = 4

    #define the extents of the grasper protraction/retraction path
    grasper_origin = x_H + np.array([[0],[0]])
    grasper_full = grasper_origin + np.array([[1],[0]])

    #define the starting position for the bottom of the grasper along this track
    x_R = grasper_origin + np.array([[0],[0]])

    #specify vectors based on the grasper in the upright position
    theta_grasper_initial = math.pi/2

    #specify the grasper radius
    r_grasper = 1
    grasper_offset = 1

    #define the positions of the I2 muscle origins
    x_I2_Borigin = grasper_origin + np.array([[0],[0]])
    x_I2_Aorigin = grasper_origin + np.array([[0],[2*r_grasper]])

    #define the position of the hinge origin
    x_hinge_origin = grasper_origin + np.array([[0],[0]])

    #specify the angle relative to horizontal for each of the attachment points fixed on the grasper surface
    theta_s = 0
    theta_I2_A = math.pi/6
    theta_I2_B = math.pi/6
    
    #plot line representing ground
    line_ground =[]
    #plot a dot at the origin
    dot_ground =[]
    #plot line representing force transducer
    line_transducer =[]
    #plot line representing track
    line_grapser_track =[]
    #plot line from R to G
    line_RG =[]
    #plot dot at point R
    dot_R =[]
    #plot dot at point G
    dot_G =[]
    #plot dot at point S
    dot_S =[]
    #plot dot at I2 attachment point A
    dot_I2_A =[]
    #plot dot at I2 attachment point B
    dot_I2_B =[]
    #plot dot at I2 attachment point A
    dot_I2_Aorigin =[]
    #plot dot at I2 attachment point B
    dot_I2_Borigin =[]
    #draw grasper
    draw_circle =[]

    #draw head
    head =[]
    dot_H_spring =[]
    #draw head spring as dashed line
    line_H_spring =[]
    #draw grasper to head spring as dashed line
    line_G_spring =[]
    
    
    def __init__(self):
        ## Preallocate arrays
        t=np.arange(self.StartingTime,self.EndTime,self.TimeStep)
        nt=len(t) # number of time points
        
        #neural state variables
        self.MCC = np.ones((1,nt))
        self.CBI2 = np.zeros((1,nt))
        self.CBI3 = np.zeros((1,nt))
        self.CBI4 = np.zeros((1,nt))
        self.B64 = np.zeros((1,nt))
        self.B4B5 = np.zeros((1,nt))
        self.B40B30 = np.zeros((1,nt))
        self.B31B32 = np.zeros((1,nt))
        self.B6B9B3 = np.zeros((1,nt))
        self.B8 = np.zeros((1,nt))
        self.B7 = np.zeros((1,nt))
        self.B38 = np.zeros((1,nt))
        self.B20 = np.zeros((1,nt))

        #muscle state variables
        self.P_I4 = np.zeros((1,nt))
        self.A_I4 = np.zeros((1,nt))
        self.P_I3_anterior = np.zeros((1,nt))
        self.A_I3_anterior = np.zeros((1,nt))
        self.T_I3 = np.zeros((1,nt))
        self.A_I3 = np.zeros((1,nt))
        self.T_I2 = np.zeros((1,nt))
        self.A_I2 = np.zeros((1,nt))
        self.T_hinge = np.zeros((1,nt))
        self.A_hinge = np.zeros((1,nt))

        #body state variables
        self.x_h = np.zeros((1,nt))
        self.x_g = np.zeros((1,nt))
        self.grasper_friction_state = np.zeros((1,nt))
        self.jaw_friction_state = np.zeros((1,nt))
        self.theta_g = np.zeros((1,nt))

        #environment variables
        self.force_on_object = np.zeros((1,nt))


        #Specify initial conditions

        self.MCC[0,0] = 1
        self.CBI2[0,0] = 1
        self.CBI3[0,0] = 0
        self.CBI4[0,0] = 0
        self.B64[0,0] = 0
        self.B4B5[0,0] = 0
        self.B40B30[0,0] = 0
        self.B31B32[0,0] = 1
        self.B6B9B3[0,0] = 0
        self.B8[0,0] = 0
        self.B7[0,0] = 0
        self.B38[0,0] = 1
        self.B20[0,0] = 0


        self.P_I4[0,0] = 0
        self.A_I4[0,0] = 0.05
        self.P_I3_anterior[0,0] = 0
        self.A_I3_anterior[0,0] = 0.05
        self.T_I3[0,0] = 0.05
        self.A_I3[0,0] = 0.05
        self.T_I2[0,0] = 0.05
        self.A_I2[0,0] = 0.05
        self.T_hinge[0,0] = 0
        self.A_hinge[0,0] = 0.05


        self.x_h[0,0] = 0
        self.x_g[0,0] = 0.1
        self.grasper_friction_state[0,0] = 0
        self.jaw_friction_state[0,0] = 0
        self.force_on_object[0,0] = 0

        #initialize electrodes to zero
        self.stim_B4B5 = np.zeros((1,nt))
        self.stim_CBI2 = np.zeros((1,nt))
        
    def RunSimulation(self):
        t=np.arange(self.StartingTime,self.EndTime,self.TimeStep)
        nt=len(t) # number of time points
        print(nt)
        self.CBI3_stimON = 0
        self.CBI3_stimOFF = 0
        self.CBI3_refractory = 0
        self.B40B30_offTime = 0
                    
        for j in range(nt-1):
            x_gh = self.x_g[0,j]-self.x_h[0,j]

            
            if (self.neuralModel == 1):
                self.NeuralModel_001(j,x_gh)
            
            if(self.muscleModel == 1):
                self.MuscleActivations_001(j)
            
            if (self.biomechanicsModel == 1):
                self.Biomechanics_001(j)
                
    
    def MuscleActivations_001(self,j):
        ## Update I4: If food present, and grasper closed, then approaches
        # pmax pressure as dp/dt=(B8*pmax-p)/tau_p.  Use a quasi-backward-Euler
        self.P_I4[0,j+1]=((self.tau_I4*self.P_I4[0,j]+self.A_I4[0,j]*self.TimeStep)/(self.tau_I4+self.TimeStep))#old -- keep this version
        self.A_I4[0,j+1]=((self.tau_I4*self.A_I4[0,j]+self.B8[0,j]*self.TimeStep)/(self.tau_I4+self.TimeStep))

        ## Update pinch force: If food present, and grasper closed, then approaches
        # pmax pressure as dp/dt=(B8*pmax-p)/tau_p.  Use a quasi-backward-Euler
        self.P_I3_anterior[0,j+1]=(self.tau_I3anterior*self.P_I3_anterior[0,j]+self.A_I3_anterior[0,j]*self.TimeStep)/(self.tau_I3anterior+self.TimeStep)
        self.A_I3_anterior[0,j+1]=(self.tau_I3anterior*self.A_I3_anterior[0,j]+(self.B38[0,j]+self.B6B9B3[0,j])*self.TimeStep)/(self.tau_I3anterior+self.TimeStep)

        ## Update I3 (retractor) activation: dm/dt=(B6-m)/tau_m
        self.T_I3[0,j+1]=(self.tau_I3*self.T_I3[0,j]+self.TimeStep*self.A_I3[0,j])/(self.tau_I3+self.TimeStep)
        self.A_I3[0,j+1]=(self.tau_I3*self.A_I3[0,j]+self.TimeStep*self.B6B9B3[0,j])/(self.tau_I3+self.TimeStep)

        ## Update I2 (protractor) activation: dm/dt=(B31-m)/tau_m.  quasi-B-Eul.
        self.T_I2[0,j+1]=((self.tau_I2_ingestion*self.CBI3[0,j]+self.tau_I2_egestion*(1-self.CBI3[0,j]))*self.T_I2[0,j]+self.TimeStep*self.A_I2[0,j])/((self.tau_I2_ingestion*self.CBI3[0,j]+self.tau_I2_egestion*(1-self.CBI3[0,j]))+self.TimeStep)
        self.A_I2[0,j+1]=((self.tau_I2_ingestion*self.CBI3[0,j]+self.tau_I2_egestion*(1-self.CBI3[0,j]))*self.A_I2[0,j]+self.TimeStep*self.B31B32[0,j])/((self.tau_I2_ingestion*self.CBI3[0,j]+self.tau_I2_egestion*(1-self.CBI3[0,j]))+self.TimeStep)

        ## Update Hinge activation: dm/dt=(B7-m)/tau_m.  quasi-B-Eul.
        #bvec(12,j+1)=(tau_m*hinge_last+dt*B7_last)/(tau_m+dt)#old
        self.T_hinge[0,j+1]=(self.tau_hinge*self.T_hinge[0,j]+self.TimeStep*self.A_hinge[0,j])/(self.tau_hinge+self.TimeStep)#new
        self.A_hinge[0,j+1]=(self.tau_hinge*self.A_hinge[0,j]+self.TimeStep*self.B7[0,j])/(self.tau_hinge+self.TimeStep)


    def NeuralModel_001(self,j,x_gh):
        ## Initialize internal tracking variables
        
        ## Update Metacerebral cell: 
        # assume here feeding arousal continues
        # indefinitely, once started. 
        """
        MCC is active IF
            General Food arousal is on
        """
        self.MCC[0,j+1]=self.MCC[0,j]

        ## Update CBI-2
        """
        CBI2 is active IF
            MCC is on 
            and (
                (Mechanical Stimulation at Lips and Chemical Stimulation at Lips and No mechanical stimuli in grasper)
                OR 
                (Mechanical in grasper and no Chemical Stimulation at Lips)
                OR
                (B4/B5 is firing strongly (>=2)))
        """

        #CBI2 - updated 6/7/2020
        #with hypothesized connections from B4/B5
        if (self.use_hypothesized_connections ==1):
            self.CBI2[0,j+1] = ((self.stim_CBI2[0,j]==0)*
                              self.MCC[0,j]*(not self.B64[0,j])*((self.sens_mechanical_lips[0,j] and self.sens_chemical_lips[0,j] and(not self.sens_mechanical_grasper[0,j]))or(self.sens_mechanical_grasper[0,j] and(not self.sens_chemical_lips[0,j]))or(self.B4B5[0,j]>=2))+
                              (self.stim_CBI2[0,j]==1))
        else:
            self.CBI2[0,j+1] = ((self.stim_CBI2[0,j]==0)*
                              self.MCC[0,j]*(not self.B64[0,j])*((self.sens_mechanical_lips[0,j] and 
                                                                  self.sens_chemical_lips[0,j] and 
                                                                  (not self.sens_mechanical_grasper[0,j])) or 
                                                                 (self.sens_mechanical_grasper[0,j] and 
                                                                  (not self.sens_chemical_lips[0,j])))+
                              (self.stim_CBI2[0,j]==1))


        ## Update CBI-3
        # requires stimuli_mech_last and stimuli_chem_last
        """
        CBI3 is active IF
            MCC is on
            and
            Mechanical Simulation at Lips
            and
            Chemical Stimulation at Lips
            and
            B4/B5 is NOT firing strongly
            and
            CBI3 is NOT in a refractory period
        """

        #CBI3 can experieince a refractory period following strong inhibition from B4/B5
        #check if a refractory period is occuring

        #modified to only turn on refreactory after the strong stimulation
        if((self.B4B5[0,j]>=2) and (self.CBI3_stimON==0)):
            self.CBI3_stimON = j   
            #self.CBI3_refractory = 1

        if ((self.CBI3_stimON !=0) and (self.B4B5[0,j]<2)):
            self.CBI3_refractory = 1
            self.CBI3_stimOFF = j  
            self.CBI3_stimON = 0    


        if(self.CBI3_refractory and j<(self.CBI3_stimOFF+self.refractory_CBI3/1000/self.TimeStep)):
            self.CBI3_refractory = 1 
        else:
            self.CBI3_stimOFF = 0
            self.CBI3_refractory = 0 



        #CBI3 - updated 6/7/2020    
        #with hypothesized connections from B4/B5    
        if (self.use_hypothesized_connections ==1):
            self.CBI3[0,j+1] = self.MCC[0,j]*(self.sens_mechanical_lips[0,j]*self.sens_chemical_lips[0,j])*((self.B4B5[0,j]<2))*(not self.CBI3_refractory)   
        else:
        #without hypothesized connections from B4/B5  
            self.CBI3[0,j+1] = self.MCC[0,j]*(self.sens_mechanical_lips[0,j]*self.sens_chemical_lips[0,j]) 




        ## Update CBI4 - added 2/27/2020
        """
        CBI4 is active IF – mediates swallowing and rejection
            MCC is on
            and
                (Mechanical Stimulation at Lips
                OR
                Chemical Stimulation at Lips)
            and
            Mechanical Stimulation in grasper
        """
        self.CBI4[0,j+1] = self.MCC[0,j]*(self.sens_mechanical_lips[0,j] or self.sens_chemical_lips[0,j])*self.sens_mechanical_grasper[0,j]

        ## Update B64
        # list of inputs
        # Protraction threshold excites
        # grasper pressure excites - still figuring out how to implement
        # retraction threshold inhibits
        # B31/32 inhibits

        #If there is mechanical and chemical stimuli at the lips and there is
        #seaweed in the grasper -> swallow

        #If there is mechanical and chemical stimuli at the lips and there is
        #NOT seaweed in the grasper -> bite

        #If there is not chemical stimuli at the lips but there is mechanical
        #stimuli ->reject

        """
        B64 is active IF
            MCC is on
            and
            IF CBI3 is active (ingestion)
                IF mechanical stimulation is in grasper (swallowing)
                    Relative Grasper Position is Greater than B64 Swallowing Protraction threshold
                IF mechanical stimulation is NOT in grasper (biting)
                    Relative Grasper Position is Greater than B64 Biting Protraction threshold
            IF CBI3 is NOT active (rejection)
                Relative Grasper Position is Greater than B64 Rejection Protraction threshold
            and
            B31/B32 is NOT active
            and
            IF CBI3 is active (ingestion)
                IF mechanical stimulation is in grasper (swallowing)
                    NOT (Relative Grasper Position is less than B64 Swallow Retraction threshold)
                IF mechanical stimulation is NOT in grasper (biting)
                    NOT(Relative Grasper Position is less than B64 Biting Retraction threshold)
            IF CBI3 is NOT active (rejection)
                NOT(Relative Grasper Position is less than B64 Rejection Retraction threshold)
        """

        B64_proprioception = ((self.CBI3[0,j]*((self.sens_mechanical_grasper[0,j] *(x_gh>self.thresh_B64_swallow_protract)) or
                                             ((not self.sens_mechanical_grasper[0,j])*(x_gh>self.thresh_B64_bite_protract)))) or
                              ((not self.CBI3[0,j])*(x_gh>self.thresh_B64_reject_protract)))

        #B64
        self.B64[0,j+1]=self.MCC[0,j]*(not self.B31B32[0,j])*B64_proprioception

        ## Update B4/B5: 
        """
        B4/B5 is active IF
            MCC is ON
            and
            IF stimulating electrode is off
                Strongly firing IF CBI3 is NOT active (rejection)
                    and
                    B64 is active (retraction)
                    and
                    Relative grasper position is greater than B4/B5 threshold (early retraction)
                weakly firing IF CBI3 is active (ingestion)
                    and
                    B64 is active (retraction)ste
                    and
                    mechanical stimulation is in grasper (swallowing)
            If stimulating electrode is on
                Activity is set to 2 to designate strong firing
        """

        #B4B5
        self.B4B5[0,j+1]= (self.MCC[0,j]*
                         ((not self.stim_B4B5[0,j])*
                          (2*(not self.CBI3[0,j])*
                           self.B64[0,j]*(x_gh>self.thresh_B4B5_protract)) +
                          ((self.CBI3[0,j])*(self.sens_mechanical_grasper[0,j])*self.B64[0,j]))
                         +2*self.stim_B4B5[0,j])

        ## Update B20 - updated 2/27/2020
        # Not active if CB1-3 is on (strongly inhibited)
        #excited by CBI2 but excitation is weaker than inhibition from CBI3
        """
        (CBI2 is active
            OR	
            CBI4 is active
            OR
            B63 (B31/32) is active)
                and
                CBI3 is NOT active
                and
                B64 is NOT active

        """
        self.B20[0,j+1] = self.MCC[0,j]*((self.CBI2[0,j] or self.CBI4[0,j]) or self.B31B32[0,j])*(not self.CBI3[0,j])*(not self.B64[0,j])

        ## Update B40/B30
        """
        B40/B30 is active IF
            MCC is ON
            and
            (CBI2 is active
            OR 
            CBI4 is active
            OR 
            B63 (B31/32) is active)
            and 
            B64 is not active
        """

        # B30/B40 have a fast inhibitory and slow excitatory connection with
        # B8a/b. To accomodate this, we track when B30/B40 goes from a active
        # state to a quiescent state 

        self.B40B30[0,j+1] = self.MCC[0,j]*((self.CBI2[0,j] or self.CBI4[0,j]) or self.B31B32[0,j])*(not self.B64[0,j])

        #check if B30/B40 has gone from active to quiescent
        if((self.B40B30[0,j] ==1) and (self.B40B30[0,j+1]==0)):
            self.B40B30_offTime = j


        ## Update B31/B32: -updated 2/27/2020
        # activated if grasper retracted enough, inhibited if
        # pressure exceeds a threshold or grasper is protracted far enough
        """
        B31/B32 is active IF
            MCC is ON
            and
            IF CBI3 is active (ingestion)
                B64 is NOT active (protraction)
                and
                    Graper pressure is less than half of its maximum (open)
                    OR
                    CBI2 is active (biting)
                and
                IF B31/B32 is NOT firing (switching to protraction)
                    The relative grasper position is less than the retraction threshold
                IF B31/B32 is firing (protraction)
                    The relative grasper position is less than the protraction threshold
            IF CBI3 is NOT active (rejection)
                B64 is NOT active (protraction)
                and
                Grasper Pressure is greater than a quarter of the maximum (closing or closed)
                and
                    CBI2 is active
                    OR
                    CBI4 is active
                and
                IF B31/B32 is NOT firing (switching to protraction)
                    The relative grasper position is less than the retraction threshold
                IF B31/B32 is firing (protraction)
                    The relative grasper position is less than the protraction threshold
        """

        #B31/B32s thresholds may vary for different behaviors. These are set
        #here
        if (self.sens_mechanical_grasper[0,j] and self.CBI3[0,j]): #swallowing
            on_thresh = self.thresh_B31_swallow_on
            off_thresh = self.thresh_B31_swallow_off
        elif (self.sens_mechanical_grasper[0,j] and (not self.CBI3[0,j])): #rejection
            on_thresh = self.thresh_B31_reject_on
            off_thresh = self.thresh_B31_reject_off
        else: #biting
            on_thresh = self.thresh_B31_bite_on
            off_thresh = self.thresh_B31_bite_off        


        self.B31B32[0,j+1]=(self.MCC[0,j]*(
            self.CBI3[0,j]*
            ((not self.B64[0,j])*((self.P_I4[0,j]<(1/2)) or self.CBI2[0,j])*
             ((not self.B31B32[0,j])*(x_gh<off_thresh)+
              self.B31B32[0,j] *(x_gh<on_thresh)))+
            (not self.CBI3[0,j])*
            ((not self.B64[0,j])*(self.P_I4[0,j]>(1/4))*(self.CBI2[0,j] or self.CBI4[0,j])*
             ((not self.B31B32[0,j])*(x_gh<off_thresh)+
              self.B31B32[0,j] *(x_gh<on_thresh)))))

        ## Update B6/B9/B3: 
        # activate once pressure is high enough in ingestion, or low enough in
        # rejection
        """
        NEW VERSION:
        B6/B9/B3 is active IF
            MCC is active
            and
            B4/B5 is NOT firing strongly
            and
            IF CBI3 is active (ingestion)
                B64 is active (retraction)
                and
                Grasper pressure is greater than B6/B3/B9 pressure threshold (closed)
            IF CBI3 is not active (rejection)
                B64 is active (retraction)
                and
                Grasper pressure is less than B6/B3/B9 pressure threshold (open)
        """
        """
        B6/B9/B3 is active IF
            MCC is active
            and
            B64 is active (retraction)
            and
            B4/B5 is NOT firing strongly
            and (
            (CBI3 is active (ingestion)
             and
             There is NOT mechanical stimulation in mouth (biting)
             and
             Grasper pressure is greater than B6/B3/B9 biting pressure
             threshold (closed))
            OR
            (CBI3 is active (ingestion)
             and
             There is mechanical stimulation in mouth (swallowing)
             and
             Grasper pressure is greater than B6/B3/B9 swallowing pressure
             threshold (closed))
            OR
            (CBI3 is NOT active (rejection)
             and
             Grasper pressure is NOT greater than B6/B3/B9 rejection pressure
             threshold (open))
            )
        """

        #B6/B9/B3
        self.B6B9B3[0,j+1]= (self.MCC[0,j]*self.B64[0,j]*(not (self.B4B5[0,j]>=2))*(
            (self.CBI3[0,j] and not self.sens_mechanical_grasper[0,j])*
            (self.P_I4[0,j]>self.thresh_B6B9B3_bite_pressure)
            +
            (self.CBI3[0,j] and self.sens_mechanical_grasper[0,j])*
            (self.P_I4[0,j]>self.thresh_B6B9B3_swallow_pressure)
            +
            (not self.CBI3[0,j])*
            (not (self.P_I4[0,j]>self.thresh_B6B9B3_reject_pressure))))


        ## Update B8a/b
        # active if excitation from B6/B9/B3 plus protracted
        # sensory feedback exceeds threshold of 1.9, and not inhibited by
        # either B31/B32 or by sensory feedback from being retracted. If B4/B5 is
        # highly excited (activation level is 2 instead of just 1) then shut
        # down B8a/b.
        """
        B8a/b is active IF
            MCC is on
            and
                B64 is active
                OR
                B40/B30 is NOT active
                OR
                B20 is active
            and
            B4/B5 is not active
            and
                B20 is active
                OR
                B31/B32 is NOT active
        """

        #B8a/b recieves slow exitatory input from B30/B40 functionally this
        #causes strong firing immediatly following B30/B40 cessation in biting
        #and swallowing
        if(self.B40B30[0,j]==0 and j<(self.B40B30_offTime+self.postActivityExcitation_B40B30/1000/self.TimeStep)):
            B40B30_excite = 1 
        else:
            B40B30_excite = 0 


        #B8a/b - updated 5/25/2020   
        self.B8[0,j+1] = (self.MCC[0,j]*(not (self.B4B5[0,j]>=2))*(
            self.CBI3[0,j]*(
                self.B20[0,j] or (B40B30_excite)*(not self.B31B32[0,j]))+
            (not self.CBI3[0,j])*(
                self.B20[0,j])))

        ## Update B7 - ONLY ACTIVE DURING EGESTION and BITING
        # turn on as you get to peak protraction
        #in biting this has a threshold that it stops applying effective force -
        #biomechanics
        """
        B7 is active IF
            ((CBI3 is NOT active (rejection)
            OR
            There is mechanical stimulation in mouth
            and
                (The relative position of the grasper is greater than the rejection protraction threshold
                OR
                Grasper pressure is very high) (closed))
            OR
            (CBI3 is active
            and
            There is NOT mechanical stimulation in mouth (biting)
            and
                (The relative position of the grasper is greater than the bite protraction threshold
                OR
                Grasper pressure is very high) (closed))
        """
        self.B7[0,j+1] = (self.MCC[0,j]*(
            (((not self.CBI3[0,j]) or  (self.sens_mechanical_grasper[0,j]))*((x_gh>=self.thresh_B7_reject_protract)or(self.P_I4[0,j]>(.97)))) +
            ((self.CBI3[0,j]  and (not self.sens_mechanical_grasper[0,j]))*((x_gh>=self.thresh_B7_bite_protract)  or(self.P_I4[0,j]>(.97))))))

        ## Update B38: 
        # If already active, remain active until protracted past
        # 0.5.  If inactive, become active if retracted to 0.1 or further. 
        """
        B38 is active IF
            MCC is ON
            and
            mechanical stimulation in the grasper (swallowing or rejection)
            and
            IF CBI3 is active (ingestion)
                Relative grasper position is less than B38 ingestion threshold
            IF CBI3 is not active (rejection)
                Turn off B38
        """

        #B38

        self.B38[0,j+1]=(self.MCC[0,j]*(self.sens_mechanical_grasper[0,j])*(
            (self.CBI3[0,j])*(((x_gh)<self.thresh_B38_retract))))

    
    def Biomechanics_001(self,j):
        ## Biomechanics
        unbroken = 1 #tracking variable to keep track of seaweed being broken off during feeding
        x_gh = self.x_g[0,j]-self.x_h[0,j]

        ## Grasper Forces
        #all forces in form F = Ax+b
        x_vec = np.array([[self.x_h[0,j]],[self.x_g[0,j]]])

        F_I2 = self.max_I2*self.T_I2[0,j]*np.dot(np.array([1,-1]),x_vec) + self.max_I2*self.T_I2[0,j]*1 #FI2 = FI2_max*T_I2*(1-(xg-xh))
        F_I3 = self.max_I3*self.T_I3[0,j]*np.dot(np.array([-1,1]),x_vec)-self.max_I3*self.T_I3[0,j]*0 #FI3 = FI3_max*T_I3*((xg-xh)-0)
        F_hinge = (x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*np.dot(np.array([-1,1]),x_vec)-(x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*0.5 #F_hinge = [hinge stretched]*F_hinge_Max*T_hinge*((xg-xh)-0.5)
        F_sp_g = self.K_sp_g*np.dot(np.array([1,-1]),x_vec)+self.K_sp_g*self.x_gh_ref #F_sp,g = K_g((xghref-(xg-xh))

        F_I4 = self.max_I4*self.P_I4[0,j]
        F_I3_ant = (self.max_I3ant*self.P_I3_anterior[0,j]*np.dot(np.array([1,-1]),x_vec)+self.max_I3ant*
                    self.P_I3_anterior[0,j]*1)#: pinch force

        #calculate F_f for grasper
        if(self.fixation_type[0,j] == 0): #object is not fixed to a contrained surface
            #F_g = F_I2+F_sp_g-F_I3-F_hinge #if the object is unconstrained it does not apply a resistive force back on the grasper. Therefore the force is just due to the muscles

            A2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*np.array([1,-1])+self.K_sp_g*np.array([1,-1])
                              -self.max_I3*self.T_I3[0,j]*np.array([-1,1])-self.max_hinge*self.T_hinge[0,j]*
                              (x_gh>0.5)*np.array([-1,1])))
            B2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3[0,j]*
                              0+(x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*0.5))
            A21 = A2[0]
            A22 = A2[1]

            #the force on the object is approximated based on the friction
            if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                #disp('static')
                F_f_g = -self.sens_mechanical_grasper[0,j]*(F_I2+F_sp_g-F_I3-F_hinge)
                self.grasper_friction_state[0,j+1] = 1
            else:
                #disp('kinetic')
                F_f_g = self.sens_mechanical_grasper[0,j]*self.mu_k_g*F_I4
                #specify sign of friction force
                F_f_g = -(F_I2+F_sp_g-F_I3-F_hinge)/abs(F_I2+F_sp_g-F_I3-F_hinge)*F_f_g
                self.grasper_friction_state[0,j+1] = 0


        elif (self.fixation_type[0,j] == 1): #object is fixed to a contrained surface
            if unbroken:
                if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                    #disp('static')
                    F_f_g = -self.sens_mechanical_grasper[0,j]*(F_I2+F_sp_g-F_I3-F_hinge)

                    #F_g = F_I2+F_sp_g-F_I3-F_hinge + F_f_g
                    self.grasper_friction_state[0,j+1] = 1

                    #identify matrix components for semi-implicit integration
                    A21 = 0
                    A22 = 0
                    B2 = 0

                else:
                    #disp('kinetic')
                    F_f_g = -np.sign(F_I2+F_sp_g-F_I3-F_hinge)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_g*F_I4

                    #specify sign of friction force
                    #F_g = F_I2+F_sp_g-F_I3-F_hinge + F_f_g
                    self.grasper_friction_state[0,j+1] = 0


                    #identify matrix components for semi-implicit integration
                    A2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*np.array([1,-1])+self.K_sp_g*np.array([1,-1])
                                      -self.max_I3*self.T_I3[0,j]*np.array([-1,1])-self.max_hinge*self.T_hinge[0,j]*
                                      (x_gh>0.5)*np.array([-1,1])))
                    B2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3[0,j]
                                      *0+(x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*0.5+F_f_g))

                    A21 = A2[0]
                    A22 = A2[1]


            else:
                #F_g = F_I2+F_sp_g-F_I3-F_hinge #if the object is unconstrained it does not apply a resistive force back on the grasper. Therefore the force is just due to the muscles

                A2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*np.array([1,-1])+self.K_sp_g*np.array([1,-1])-self.max_I3
                                  *self.T_I3[0,j]*np.array([-1,1])-self.max_hinge*self.T_hinge[0,j]*(x_gh>0.5)
                                  *np.array([-1,1])))
                B2 = (1/self.c_g*(self.max_I2*self.T_I2[0,j]*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3[0,j]*
                                  0+(x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*0.5))


                A21 = A2[0]
                A22 = A2[1]

                #the force on the object is approximated based on the friction
                if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                    #disp('static')
                    F_f_g = -self.sens_mechanical_grasper[0,j]*(F_I2+F_sp_g-F_I3-F_hinge)
                    self.grasper_friction_state[0,j+1] = 1
                else:
                    #disp('kinetic')
                    F_f_g = self.sens_mechanical_grasper[0,j]*self.mu_k_g*F_I4
                    #specify sign of friction force
                    F_f_g = -(F_I2+F_sp_g-F_I3-F_hinge)/abs(F_I2+F_sp_g-F_I3-F_hinge)*F_f_g
                    self.grasper_friction_state[0,j+1] = 0



        #[j*dt position_grasper_relative I2 F_sp I3 hinge GrapserPressure_last F_g]

        ## Body Forces
        #all forces in the form F = Ax+b
        F_sp_h = self.K_sp_h*np.dot(np.array([-1,0]),x_vec)+self.x_h_ref*self.K_sp_h
        #all muscle forces are equal and opposite
        if(self.fixation_type[0,j] == 0):     #object is not constrained
            #F_h = F_sp_h #If the object is unconstrained it does not apply a force back on the head. Therefore the force is just due to the head spring.

            A1 = 1/self.c_h*self.K_sp_h*np.array([-1,0])
            B1 = 1/self.c_h*self.x_h_ref*self.K_sp_h


            A11 = A1[0]
            A12 = A1[1]

            if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                #disp('static2')
                F_f_h = -self.sens_mechanical_grasper[0,j]*(F_sp_h+F_f_g) #only calculate the force if an object is actually present
                self.jaw_friction_state[0,j+1] = 1
            else:
                #disp('kinetic2')
                F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_h*F_I3_ant #only calculate the force if an object is actually present
                self.jaw_friction_state[0,j+1] = 0

        elif (self.fixation_type[0,j] == 1):
            #calcuate friction due to jaws
            if unbroken: #if the seaweed is intact
                if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                    #disp('static2')
                    F_f_h = -self.sens_mechanical_grasper[0,j]*(F_sp_h+F_f_g) #only calculate the force if an object is actually present
                    #F_h = F_sp_h+F_f_g + F_f_h
                    self.jaw_friction_state[0,j+1] = 1

                    A11 = 0
                    A12 = 0
                    B1 = 0

                else:
                    #disp('kinetic2')
                    F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_h*F_I3_ant #only calculate the force if an object is actually present
                    #F_h = F_sp_h+F_f_g + F_f_h

                    self.jaw_friction_state[0,j+1] = 0

                    if (self.grasper_friction_state[0,j+1] == 1): #object is fixed and grasper is static  
                    # F_f_g = -mechanical_in_grasper*(F_I2+F_sp_g-F_I3-F_Hi)
                        A1 = (1/self.c_h*(self.K_sp_h*np.array([-1,0])+(-self.sens_mechanical_grasper[0,j]*
                                                                       (self.max_I2*self.T_I2[0,j]*np.array([1,-1])
                                                                        +self.K_sp_g*np.array([1,-1])-self.max_I3*
                                                                        self.T_I3[0,j]*np.array([-1,1])-self.max_hinge*
                                                                        self.T_hinge[0,j]*(x_gh>0.5)*np.array([-1,1]))
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_h*
                                                                       self.max_I3ant*self.P_I3_anterior[0,j]
                                                                       *np.array([1,-1]))))
                        B1 = (1/self.c_h*(self.x_h_ref*self.K_sp_h+(-self.sens_mechanical_grasper[0,j]*(self.max_I2*self.T_I2[0,j]*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3[0,j]*0+(x_gh>0.5)*self.max_hinge*self.T_hinge[0,j]*0.5))
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_h*self.max_I3ant*self.P_I3_anterior[0,j]*1))

                    else: #both are kinetic
                    #F_f_g = -np.sign(F_I2+F_sp_g-F_I3-F_Hi)*mechanical_in_grasper*mu_k_g*F_I4
                        A1 = (1/self.c_h*(self.K_sp_h*np.array([-1,0])-np.sign(F_sp_h+F_f_g)[0]
                                         *self.sens_mechanical_grasper[0,j]*self.mu_k_h*self.max_I3ant*
                                         self.P_I3_anterior[0,j]*np.array([1,-1])))
                        B1 = (1/self.c_h*(self.x_h_ref*self.K_sp_h-np.sign(F_I2+F_sp_g-F_I3-F_hinge)[0]*
                                          self.sens_mechanical_grasper[0,j]*self.mu_k_g*F_I4
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*
                                          self.mu_k_h*self.max_I3ant*self.P_I3_anterior[0,j]*1))               

                    A11 = A1[0]
                    A12 = A1[1]

            else: # if the seaweed is broken the jaws act as if unconstrained
                if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                #disp('static2')
                    F_f_h = -self.sens_mechanical_grasper[0,j]*(F_sp_h+F_f_g) #only calculate the force if an object is actually present
                    self.jaw_friction_state[0,j+1] = 1
                else:
                    #disp('kinetic2')
                    F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper[0,j]*self.mu_k_h*F_I3_ant #only calculate the force if an object is actually present
                    self.jaw_friction_state[0,j+1] = 0


                A1 = 1/self.c_h*self.K_sp_h*[-1,0]
                B1 = 1/self.c_h*self.x_h_ref*self.K_sp_h

                A11 = A1[0]
                A12 = A1[1]
                self.jaw_friction_state[0,j+1] = 0



        #[position_buccal_last F_h F_sp I3 hinge force_pinch F_H]


        ## Integrate body motions
        #uncomment to remove periphery
        #F_g = 0
        #F_H = 0

        A = np.array([[A11,A12],[A21,A22]])
        B = np.array([[B1],[B2]])

        x_last = np.array(x_vec)


        x_new = (1/(1-self.TimeStep*A.trace()))*(np.dot((np.identity(2)+self.TimeStep*
                                                         np.array([[-A22,A12],[A21,-A11]])),x_last)+
                                                         self.TimeStep*B)

        self.x_g[0,j+1] = x_new[1]
        self.x_h[0,j+1] = x_new[0]

        ## calculate force on object
        self.force_on_object[0,j+1] = F_f_g+F_f_h

        #check if seaweed is broken
        if (self.fixation_type[0,j] ==1):
            if (self.force_on_object[0,j+1]>self.seaweed_strength):
                unbroken = 0

            #check to see if a new cycle has started
            x_gh_next = self.x_g[0,j+1]-self.x_h[0,j+1]

            if (not unbroken and x_gh <0.3 and x_gh_next>x_gh):
                unbroken = 1 

            self.force_on_object[0,j+1]= unbroken*self.force_on_object[0,j+1]
        
    def SetBiomechanicsModel(self,model):
        self.biomechanicsModel = model
        
    def SetNeuralModel(self,model):
        self.neuralModel = model
        
    def SetMuscleModel(self,model):
        self.muscleModel = model
        
    def SetSensoryStates(self,*args):
        t=np.arange(self.StartingTime,self.EndTime,self.TimeStep)
        nt=len(t) # number of time points
        
        nargin = len(args)


        if (nargin == 1):
            behavior = args[0]
            if (behavior=='bite'):
                self.sens_chemical_lips = np.ones((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.zeros((1,nt))
                self.fixation_type = np.zeros((1,nt))
            elif (behavior=='swallow'):
                self.sens_chemical_lips = np.ones((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.ones((1,nt))
                self.fixation_type = np.ones((1,nt))
            elif (behavior=='reject'):
                self.sens_chemical_lips = np.zeros((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.ones((1,nt))
                self.fixation_type = np.zeros((1,nt))
        elif (nargin == 3):
            behavior_1 = args[0]
            behavior_2 = args[1]
            t_switch = args[2]

            step_switch = round(t_switch/self.TimeStep)

            if (behavior_1=='bite'):
                self.sens_chemical_lips = np.ones((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.zeros((1,nt))
                self.fixation_type = np.zeros((1,nt))
            elif (behavior_1=='swallow'):
                self.sens_chemical_lips = np.ones((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.ones((1,nt))
                self.fixation_type = np.ones((1,nt))
            elif (behavior_1=='reject'):
                self.sens_chemical_lips = np.zeros((1,nt))
                self.sens_mechanical_lips = np.ones((1,nt))
                self.sens_mechanical_grasper = np.ones((1,nt))
                self.fixation_type = np.zeros((1,nt))

            if (behavior_2=='bite'):
                self.sens_chemical_lips[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_lips[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_grasper[1,step_switch:nt] = np.zeros((1,len(np.range(step_switch,nt))))
                self.fixation_type = np.zeros((1,len(np.range(step_switch,nt))))
            elif (behavior_2=='reject'):
                self.sens_chemical_lips[1,step_switch:nt] = np.zeros((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_lips[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_grasper[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.fixation_type[1,step_switch:nt] = np.zeros((1,len(np.range(step_switch,nt))))
            elif (behavior_2=='swallow'):
                self.sens_chemical_lips[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_lips[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.sens_mechanical_grasper[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))
                self.fixation_type[1,step_switch:nt] = np.ones((1,len(np.range(step_switch,nt))))


    def SetStimulationTrains(self,neuron,onTime,duration):
        t=np.arange(self.StartingTime,self.EndTime,self.TimeStep)
        nt=len(t) # number of time points

        if (neuron=='B4B5'):
            self.stim_B4B5[1:nt] = np.zeros((1,nt)) # initialize extracellular stimulation of B4/B5
            self.stim_B4B5[onTime:(onTime+duration)] = ones(1,len(self.stim_B4B5[onTime:(onTime+duration)]))
            self.stim_B4B5[(onTime+duration):end] = zeros(1,len(self.stim_B4B5[(onTime+duration):end]))

        if (neuron=='CBI2'):
            self.stim_CBI2[1:nt] = np.zeros((1,nt)) # initialize extracellular stimulation of CBI-2
            self.stim_CBI2[onTime:(onTime+duration)] = np.ones((1,len(self.stim_CBI2[onTime:(onTime+duration)])))
            self.stim_CBI2[(onTime+duration):end] = np.zeros((1,len(self.stim_CBI2[(onTime+duration):end])))
            
            
    def GeneratePlots(self,label,xlimits):
        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime,self.TimeStep))

        axs = plt.figure(figsize=(10,20), constrained_layout=True).subplots(18,1)
        xl=xlimits; # show full time scale
        lineW =2


        #External Stimuli
        ax0 = axs[0]
        ax0.plot(t.transpose(),self.sens_mechanical_grasper.transpose(), color=[56/255, 232/255, 123/255],linewidth=2) #mechanical in grasper
        #set(gca,'FontSize',16)

        #set(gca,'xtick',[])
        #set(gca,'ytick',[0,1])
        #set(gca,'YTickLabel',[]);
        ax0.set_ylabel('Mech. in Grasper')
        #ylim([0 1])
        #grid on
        #xlim(xl)
        #hYLabel = get(gca,'YLabel');
        #set(hYLabel,'rotation',0,'VerticalAlignment','middle','HorizontalAlignment','right','Position',get(hYLabel,'Position')-[0.05 0 0])

        #set(gca,'XColor','none')
        i=1
        ax = axs[i]
        ax.plot(t.transpose(),self.sens_chemical_lips.transpose(), color=[70/255, 84/255, 218/255],linewidth=2) #chemical at lips
        ax.set_ylabel('Chem. at Lips')
        # ax.ylabel('Chem. at Lips',rotation=30)
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.sens_mechanical_lips.transpose(), color=[47/255, 195/255, 241/255],linewidth=2) #mechanical at lips
        ax.set_ylabel('Mech. at Lips')
        i=i+1
        
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI2.transpose(),'k',linewidth=lineW) # CBI2
        ax.set_ylabel('CBI-2')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.CBI3.transpose(),'k',linewidth=lineW) # CBI3
        ax.set_ylabel('CBI-3')
        i=i+1
       
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI4.transpose(),'k',linewidth=lineW) # CBI4
        ax.set_ylabel('CBI-4')
        i=i+1
        
        #Interneurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B64.transpose(),linewidth=lineW, color=[90/255, 131/255, 198/255]) # B64
        ax.set_ylabel('B64', color=[90/255, 131/255, 198/255])
        i=i+1


        ax = axs[i]
        ax.plot(t.transpose(),self.B20.transpose(),linewidth=lineW, color=[44/255, 166/255, 90/255]) # B20
        i=i+1;
        ax.set_ylabel('B20', color=[44/255, 166/255, 90/255])


        ax = axs[i]
        ax.plot(t.transpose(),self.B40B30.transpose(),linewidth=lineW, color=[192/255, 92/255, 185/255]) # B40/B30
        i=i+1;
        ax.set_ylabel('B40/B30', color=[192/255, 92/255, 185/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B4B5.transpose(),linewidth=lineW, color=[51/255, 185/255, 135/255]) # B4/5
        i=i+1;
        ax.set_ylabel('B4/B5', color=[51/255, 185/255, 135/255])
        

        #motor neurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B31B32.transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) # I2 input
        i=i+1;
        ax.set_ylabel('B31/B32',color=[220/255, 81/255, 81/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B8.transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) # B8a/b
        i=i+1;
        ax.set_ylabel('B8a/b', color=[213/255, 155/255, 196/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B38.transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) # B38
        i=i+1;
        ax.set_ylabel('B38', color=[238/255, 191/255, 70/255])
        

        ax = axs[i]
        ax.plot(t.transpose(),self.B6B9B3.transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) # B6/9/3
        i=i+1;
        ax.set_ylabel('B6/B9/B3', color=[90/255, 155/255, 197/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B7.transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) # B7
        i=i+1;
        ax.set_ylabel('B7', color=[56/255, 167/255, 182/255])
        
        #Grasper Motion
        grasper_motion = self.x_g - self.x_h
        ax = axs[i]
        ax.plot(t.transpose(),grasper_motion.transpose(),'b',linewidth=lineW)
        i=i+1;
        ax.set_ylabel('Grasper Motion', color=[0/255, 0/255, 255/255])
       

        #subplot(15,1,15)
        ax = axs[i]
        ax.plot(t.transpose(),self.force_on_object.transpose(),'k',linewidth=lineW)
        ax.set_ylabel('Force', color=[0/255, 0/255, 0/255])    
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(), self.grasper_friction_state.transpose(),'k',linewidth = lineW)
        ax.set_ylabel('friction state')



        
        plt.show(block=False)
    
    def GeneratePlots_WS(self,label,xlimits):
        
        import math, copy

        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime,self.TimeStep))
        tmp =  [1] * 15
        tmp.extend([2,2])
        axs = plt.figure(figsize=(10,15), constrained_layout=True).subplots(17,1, sharex=True, gridspec_kw={'height_ratios': tmp})
        xl=xlimits; # show full time scale
        lineW =2


        #External Stimuli
        ax0 = axs[0]
        ax0.plot(t.transpose(),self.sens_mechanical_grasper.transpose(), color=[56/255, 232/255, 123/255],linewidth=2) #mechanical in grasper
        #set(gca,'FontSize',16)

        #set(gca,'xtick',[])
        #set(gca,'ytick',[0,1])
        #set(gca,'YTickLabel',[]);
        ax0.set_ylabel('Mech. in Grasper')
        #ylim([0 1])
        #grid on
        #xlim(xl)
        #hYLabel = get(gca,'YLabel');
        #set(hYLabel,'rotation',0,'VerticalAlignment','middle','HorizontalAlignment','right','Position',get(hYLabel,'Position')-[0.05 0 0])

        #set(gca,'XColor','none')
        i=1
        ax = axs[i]
        ax.plot(t.transpose(),self.sens_chemical_lips.transpose(), color=[70/255, 84/255, 218/255],linewidth=2) #chemical at lips
        ax.set_ylabel('Chem. at Lips')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.sens_mechanical_lips.transpose(), color=[47/255, 195/255, 241/255],linewidth=2) #mechanical at lips
        ax.set_ylabel('Mech. at Lips')
        i=i+1
        
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI2.transpose(),'k',linewidth=lineW) # CBI2
        ax.set_ylabel('CBI-2')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.CBI3.transpose(),'k',linewidth=lineW) # CBI3
        ax.set_ylabel('CBI-3')
        i=i+1
       
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI4.transpose(),'k',linewidth=lineW) # CBI4
        ax.set_ylabel('CBI-4')
        i=i+1
        
        #Interneurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B64.transpose(),linewidth=lineW, color=[90/255, 131/255, 198/255]) # B64
        ax.set_ylabel('B64', color=[90/255, 131/255, 198/255])
        i=i+1


        ax = axs[i]
        ax.plot(t.transpose(),self.B20.transpose(),linewidth=lineW, color=[44/255, 166/255, 90/255]) # B20
        i=i+1;
        ax.set_ylabel('B20', color=[44/255, 166/255, 90/255])


        ax = axs[i]
        ax.plot(t.transpose(),self.B40B30.transpose(),linewidth=lineW, color=[192/255, 92/255, 185/255]) # B40/B30
        i=i+1;
        ax.set_ylabel('B40/B30', color=[192/255, 92/255, 185/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B4B5.transpose(),linewidth=lineW, color=[51/255, 185/255, 135/255]) # B4/5
        i=i+1;
        ax.set_ylabel('B4/B5', color=[51/255, 185/255, 135/255])
        

        #motor neurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B31B32.transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) # I2 input
        i=i+1;
        ax.set_ylabel('B31/B32',color=[220/255, 81/255, 81/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B8.transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) # B8a/b
        i=i+1;
        ax.set_ylabel('B8a/b', color=[213/255, 155/255, 196/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B38.transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) # B38
        i=i+1;
        ax.set_ylabel('B38', color=[238/255, 191/255, 70/255])
        

        ax = axs[i]
        ax.plot(t.transpose(),self.B6B9B3.transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) # B6/9/3
        i=i+1;
        ax.set_ylabel('B6/B9/B3', color=[90/255, 155/255, 197/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B7.transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) # B7
        i=i+1;
        ax.set_ylabel('B7', color=[56/255, 167/255, 182/255])
        
        #Grasper Motion plot
        grasper_motion = self.x_g - self.x_h
        ax = axs[i]
        ax.plot(t.transpose(),grasper_motion.transpose(),'b',linewidth=lineW)

        # overlay the grasper friction state as thick blue dots
        grasper_motion_gfs = copy.deepcopy(grasper_motion)
        t_gfs = copy.deepcopy(t)
        grasper_motion_gfs[self.grasper_friction_state != 1] = math.nan
        t_gfs[self.grasper_friction_state != 1] = math.nan
        ax.plot(t_gfs.transpose(), grasper_motion_gfs.transpose(),'b', linewidth = lineW * 2)
        
        # overlay b&w bars
        gm_delta = np.zeros_like(grasper_motion)
        t_delta = copy.deepcopy(t)
        gm_delta[:,1:] = grasper_motion[:,1:] - grasper_motion[:,:-1]

        t_delta[gm_delta <= 0] = math.nan
        gm_delta[gm_delta > 0] = 1.25
        gm_delta[gm_delta != 1.25 ] = math.nan

        ax.plot(t.transpose(), 1.25 * np.ones_like(t).transpose(), 'k', linewidth = lineW * 3)
        ax.plot(t_delta.transpose(), gm_delta.transpose(),'w', linewidth = lineW * 2.8)


        

        i=i+1;
        ax.set_ylabel('Grasper Motion', color=[0/255, 0/255, 255/255])
       

        #subplot(15,1,15)
        ax = axs[i]
        ax.plot(t.transpose(),self.force_on_object.transpose(),'k',linewidth=lineW)
        ax.set_ylabel('Force', color=[0/255, 0/255, 0/255])    
        # i=i+1

        # ax = axs[i]
        # ax.plot(t.transpose(), self.grasper_friction_state.transpose(),'k',linewidth = lineW)
        # ax.set_ylabel('friction state')


        for i, a in enumerate(axs):
            # a.plot(t, np.sin((i + 1) * 2 * np.pi * t))
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            if i < 15:
                a.set_ylim([0, 1])
            elif i == 15:
                a.set_ylim([0, 1.5])
        plt.xticks([])
        plt.yticks([])
        plt.setp(axs, xlim=(np.min(t),np.max(t)))
        plt.show(block=False)
    
    def RotationMatrixZ(self,theta):
        return np.matrix([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    
    def unitVec(self,origin,endpoint):
        vec = origin-endpoint;
        if (vec.item(0)==0 and vec.item(1)==0):
            return np.array([[0],[0]])
        else:
            return (origin - endpoint)/np.linalg.norm(origin - endpoint)
    
    def Animate(self,j):
        # print(j)
        lines =[]
        
        M_grasper_rot = self.RotationMatrixZ(self.theta_g[0,j])
        self.x_R = np.array([[self.x_g[0,j]],[0]])
        self.x_H = np.array([[self.x_h[0,j]],[0]])

        #define vectors to various points on grasper
        x_G_def = M_grasper_rot*np.array([[self.grasper_offset*math.cos(self.theta_grasper_initial)],
                                          [self.grasper_offset*math.sin(self.theta_grasper_initial)]]) + self.x_R
        x_S_def = x_G_def + M_grasper_rot*np.array([[self.r_grasper*math.cos(self.theta_s)],
                                                    [self.r_grasper*math.sin(self.theta_s)]])
        x_I2_A_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_A)],
                                                       [self.r_grasper*math.sin(self.theta_I2_A)]])
        x_I2_B_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_B)],
                                                       [-self.r_grasper*math.sin(self.theta_I2_B)]])

        #rotate all the vectors
        #x_G = M_grasper_rot*x_G_def
        #x_S = M_grasper_rot*x_S_def
        #x_I2_A = M_grasper_rot*x_I2_A_def
        #x_I2_B = M_grasper_rot*x_I2_B_def
        x_G = x_G_def
        x_S = x_S_def
        x_I2_A = x_I2_A_def
        x_I2_B = x_I2_B_def

        #Calculate vectors and lengths for forces/tensions

        #I2
        vec_I2_A = self.unitVec(self.x_I2_Aorigin,x_I2_A)
        vec_I2_B = self.unitVec(self.x_I2_Borigin,x_I2_B)
        length_I2 = np.linalg.norm(vec_I2_A)+np.linalg.norm(vec_I2_B)+np.linalg.norm(x_I2_A-x_I2_B)

        #I3
        vec_I3 = np.array([[-1],[0]])
        length_I3 = (x_G[0,0] -(-self.r_grasper))

        #hinge
        vec_hinge = self.unitVec(self.x_hinge_origin,self.x_R)
        length_hinge = np.linalg.norm(self.x_R - self.x_hinge_origin)

        #seaweed or tube
        vec_object = np.array([[9],[0]]) # prev: 1,0
        
        #plot slug head
        

        #plot line representing track
        self.line_grapser_track.set_data([self.grasper_origin[0],self.grasper_full[0]],[self.grasper_origin[1],self.grasper_full[1]])

        #plot line from R to G
        self.line_RG.set_data([self.x_R[0],x_G[0]],[self.x_R[1],x_G[1]])
        #plot dot at point R
        # self.dot_R.set_data(self.x_R[0],self.x_R[1])
        #plot dot at point G
        # self.dot_G.set_data(x_G[0],x_G[1])
        #plot dot at point S
        # self.dot_S.set_data(x_S[0],x_S[1])
        #plot dot at I2 attachment point A
        self.dot_I2_A.set_data(x_I2_A[0],x_I2_A[1])
        #plot dot at I2 attachment point B
        self.dot_I2_B.set_data(x_I2_B[0],x_I2_B[1])
        #plot dot at I2 attachment point A
        self.dot_I2_Aorigin.set_data(self.x_I2_Aorigin[0],self.x_I2_Aorigin[1])
        #plot dot at I2 attachment point B
        self.dot_I2_Borigin.set_data(self.x_I2_Borigin[0],self.x_I2_Borigin[1])

        #draw grasper
        self.draw_circle.center = x_G[0],x_G[1]
        self.wedge.set_center((x_G[0][0,0]  , x_G[1][0,0]))
        self.wedge.set_theta1(30*(1-self.P_I4[0,j]))
        self.wedge.set_theta2(360 - 30*(1-self.P_I4[0,j]) )

        
        #draw head
        self.head.set_xy((self.x_H[0],self.x_H[1]-self.x_H_height/2))
        
        #plot a dot at the head spring attachment
        self.dot_H_spring.set_data(self.x_H[0],self.x_H[1])

        #draw head spring as dashed line
        self.line_H_spring.set_data([self.x_H[0],self.x_ground[0]],[self.x_H[1],self.x_ground[1]])

        #draw grasper to head spring as dashed line
        self.line_G_spring.set_data([self.grasper_origin[0],self.x_H[0]],[self.grasper_origin[1],self.x_H[1]])

        lines.append(self.line_ground)
        lines.append(self.dot_ground)
        lines.append(self.line_transducer)
        lines.append(self.line_grapser_track)
        lines.append(self.line_RG)
        # lines.append(self.dot_R)
        # lines.append(self.dot_G)
        # lines.append(self.dot_S)
        lines.append(self.dot_I2_A)
        lines.append(self.dot_I2_B)
        lines.append(self.dot_I2_Aorigin)
        lines.append(self.dot_I2_Borigin)
        lines.append(self.draw_circle)
        lines.append(self.food)
        lines.append(self.wedge)


        self.ab.remove()
        sh = mpimg.imread('slug.png')
        imagebox = OffsetImage(sh, zoom=0.15)
        self.ab = AnnotationBbox(imagebox, (self.x_H[0]+0.5, 0.5), frameon=False)
        axes.add_artist(self.ab)

        lines.append(self.ab) 
        
        lines.append(self.head)
        lines.append(self.dot_H_spring)
        lines.append(self.line_H_spring)
        lines.append(self.line_G_spring)
        
        return lines

    
    def VisualizeMotion(self):
        if (self.biomechanicsModel==1):
            self.x_H = self.x_ground + np.array([[0],[0]])
            self.grasper_origin = self.x_H + np.array([[0],[0]])
            self.grasper_offset=0            
            #show the current pose of the model
            global axes
            figure, axes = plt.subplots()
            figure.set_size_inches(4.5, 3, True)
            resultShape = np.shape(self.x_h)
            print('*resultShape =', resultShape)
            
            
            M_grasper_rot = self.RotationMatrixZ(self.theta_g[0,0])
            self.x_R = np.array([[self.x_g[0,0]],[0]])
            self.x_H = np.array([[self.x_h[0,0]],[0]])

            #define vectors to various points on grasper
            x_G_def = M_grasper_rot*np.array([[self.grasper_offset*math.cos(self.theta_grasper_initial)],
                                              [self.grasper_offset*math.sin(self.theta_grasper_initial)]]) + self.x_R
            x_S_def = x_G_def + M_grasper_rot*np.array([[self.r_grasper*math.cos(self.theta_s)],
                                                        [self.r_grasper*math.sin(self.theta_s)]])
            x_I2_A_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_A)],
                                                           [self.r_grasper*math.sin(self.theta_I2_A)]])
            x_I2_B_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_B)],
                                                           [-self.r_grasper*math.sin(self.theta_I2_B)]])

            #rotate all the vectors
            #x_G = M_grasper_rot*x_G_def
            #x_S = M_grasper_rot*x_S_def
            #x_I2_A = M_grasper_rot*x_I2_A_def
            #x_I2_B = M_grasper_rot*x_I2_B_def
            x_G = x_G_def
            x_S = x_S_def
            x_I2_A = x_I2_A_def
            x_I2_B = x_I2_B_def

            #Calculate vectors and lengths for forces/tensions

            #I2
            vec_I2_A = self.unitVec(self.x_I2_Aorigin,x_I2_A)
            vec_I2_B = self.unitVec(self.x_I2_Borigin,x_I2_B)
            length_I2 = np.linalg.norm(vec_I2_A)+np.linalg.norm(vec_I2_B)+np.linalg.norm(x_I2_A-x_I2_B)

            #I3
            vec_I3 = np.array([[-1],[0]])
            length_I3 = (x_G[0,0] -(-self.r_grasper))

            #hinge
            vec_hinge = self.unitVec(self.x_hinge_origin,self.x_R)
            length_hinge = np.linalg.norm(self.x_R - self.x_hinge_origin)

            #seaweed or tube
            vec_object = np.array([[1],[0]])

            lines = []
            
            #plot line representing ground
            self.line_ground, = axes.plot([self.x_ground[0],self.x_ground[0]],[self.x_ground[1]+self.len_ground_line/2,
                                                           self.x_ground[1]-self.len_ground_line/2])
            lines.append(self.line_ground)
            #plot a dot at the origin
            self.dot_ground, = axes.plot(self.x_ground[0],self.x_ground[1],'ko', alpha = 0.05)
            lines.append(self.dot_ground)

            #plot line representing force transducer
            self.line_transducer, = axes.plot([self.x_transducer[0],self.x_transducer[0]],[self.x_transducer[1]+self.len_transducer_line/2,
                                                                   self.x_transducer[1]-self.len_transducer_line/2],'y')
            lines.append(self.line_transducer)

            
            

            #plot line representing track
            self.line_grapser_track, = axes.plot([self.grasper_origin[0],self.grasper_full[0]],[self.grasper_origin[1],self.grasper_full[1]],'y--')
            lines.append(self.line_grapser_track)
            #plot line from R to G
            self.line_RG, = axes.plot([self.x_R[0],x_G[0]],[self.x_R[1],x_G[1]])
            lines.append(self.line_RG)
            #plot dot at point R
            # self.dot_R, = axes.plot(self.x_R[0],self.x_R[1],'ko')
            # lines.append(self.dot_R)



            #plot dot at point G
            # self.dot_G, = axes.plot(x_G[0],x_G[1],'mo')
            # lines.append(self.dot_G)
            self.wedge = patches.Wedge((0,0), 0.5, 30, 300, ec="none")
            axes.add_patch(self.wedge)
            lines.append(self.wedge)

            #plot 'food'
            self.food, = axes.plot([0.5, 8],[0, 0],'g', linewidth=6, alpha = 0.75)
            lines.append(self.food)

            #plot dot at point S
            # self.dot_S, = axes.plot(x_S[0],x_S[1],'go')
            # lines.append(self.dot_S)
            #plot dot at I2 attachment point A
            self.dot_I2_A, = axes.plot(x_I2_A[0],x_I2_A[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_A)
            #plot dot at I2 attachment point B
            self.dot_I2_B, = axes.plot(x_I2_B[0],x_I2_B[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_B)
            #plot dot at I2 attachment point A
            self.dot_I2_Aorigin, = axes.plot(self.x_I2_Aorigin[0],self.x_I2_Aorigin[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_Aorigin)
            #plot dot at I2 attachment point B
            self.dot_I2_Borigin, = axes.plot(self.x_I2_Borigin[0],self.x_I2_Borigin[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_Borigin)
            #draw grasper
            self.draw_circle = plt.Circle(x_G, self.r_grasper,fill=False,color='blue', alpha = 0.05)
            axes.add_artist(self.draw_circle)
            axes.set_xlim([-2,6])
            axes.set_ylim([-3,3])
            axes.set_aspect(aspect = 1)
            lines.append(self.draw_circle)
            #draw head
            self.head = patches.Rectangle((self.x_H[0],self.x_H[1]-self.x_H_height/2),self.x_H_width,self.x_H_height, alpha = 0.05)
            self.head.set_fill(0)
            axes.add_patch(self.head)
            lines.append(self.head)
       
            sh = mpimg.imread('slug.png')
            imagebox = OffsetImage(sh, zoom=0.02)
            self.ab = AnnotationBbox(imagebox, (2, 0), frameon=False)
            global tmp
            tmp = axes.add_artist(self.ab)
            lines.append(self.ab) 

            # artists.append(ax.add_artist(ab))

            #plot a dot at the head spring attachment
            self.dot_H_spring, = axes.plot(self.x_H[0],self.x_H[1],'ko', alpha = 0.05)
            lines.append(self.dot_H_spring)

            #draw head spring as dashed line
            self.line_H_spring, = axes.plot([self.x_H[0],self.x_ground[0]],[self.x_H[1],self.x_ground[1]],'k--', alpha = 0.05)

            lines.append(self.line_H_spring)
            #draw grasper to head spring as dashed line
            self.line_G_spring, = axes.plot([self.grasper_origin[0],self.x_H[0]],[self.grasper_origin[1],self.x_H[1]],'b--', alpha = 0.05)
            lines.append(self.line_G_spring)
            dpi = 600
            ani = animation.FuncAnimation(figure, self.Animate, resultShape[1], interval=50, blit=False)
            # writer = animation.PillowWriter(fps=20)  
            # ani.save("dummy_blitoff.gif", writer=writer, dpi = dpi) 

            writer = animation.writers['ffmpeg'](fps=20)
            ani.save('test.mp4',writer=writer,dpi=dpi)
            # plt.show()


