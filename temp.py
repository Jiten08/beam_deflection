import numpy as np
import plotly.graph_objects as go
import plotly.io as pio  


pointLoads = np.array([[]])  #PointForces [location, xmag, ymag]

#Inputs
span = 17 #span of the beam
A = 3     #Distance to the left support
B = 13    #Distance to the rigth support 

#ForceData
pointLoads = np.array([[6, 0, -90]]) #[location, xmag, ymag]
distributedLoads = np.array([[8, 17, -10]]) #[xStart, xEnd, mag]

#Defaults 
delta = 0.05  #Distance between each data point
X = np.arange(0, span+delta, delta) #Range of X-coordinates
nUDL = len(distributedLoads[0]) #test for uniformly distributed loads


reactions = np.array([0.0, 0, 0])  #Reactions (Va, Ha, Vb)
shearForce = np.zeros(len(X))  #Shear forces at each data point
bendingMoment = np.zeros(len(X))  #Bending moment at each data point 

def reactions_PL(n):
    xp = pointLoads[n][0] #Location of point load
    fx = pointLoads[n][1] #Horizontal component
    fy = pointLoads[n][2] #Vertical component

    la_p = A - xp #lever arm at the pointLoad about A
    mp = fy*la_p  #moment generated by pointLoad about A
    la_vb = B - A #lever arm of vertical reaction at B about A

    Vb = mp/la_vb #Vertical reaction at B
    Va = -fy -Vb  #Vertical reaction at A
    Ha = -fx      #Horizontal reaction at A

    return Va, Vb, Ha

def reactions_UDL(n):
    xStart = distributedLoads[n][0]
    xEnd = distributedLoads[n][1]
    fy = distributedLoads[n][2]

    fy_Res = fy*(xEnd-xStart)
    x_Res = xStart + 0.5*(xEnd-xStart)

    la_p = A - x_Res #lever arm at the resultant pointLoad about A
    mp = fy_Res*la_p  #moment generated by resultant pointLoad about A
    la_vb = B - A #lever arm of vertical reaction at B about A

    Vb = mp/la_vb #Vertical reaction at B
    Va = -fy_Res - Vb  #Vertical reaction at A

    return Va, Vb


def shear_moment_PL(n):
    xp = pointLoads[n][0]
    fy = pointLoads[n][2]
    Va = PL_record[n][0]
    Vb = PL_record[n][2]

    Shear = np.zeros(len(X))
    Moment = np.zeros(len(X))

    for i, x in enumerate(X):
        shear = 0 
        moment = 0

        if x>A:
            #Calculate shear and moment due to reaction at A
            shear += Va
            moment -= Va*(x-A)
        
        if x>B:
            #Calculate shear and moment due to reaction at B
            shear += Vb
            moment -= Vb*(x-B)

        if x>xp:
            #Calculate shear and moment due to point load 
            shear += fy
            moment -= fy*(x-xp)

        Shear[i] = shear
        Moment[i] = moment

    return Shear, Moment

def shear_moment_UDL(n):
     xStart = distributedLoads[n][0]
     xEnd = distributedLoads[n][1]
     fy = distributedLoads[n][2]
     Va = UDL_record[n][0]
     Vb = UDL_record[n][1]

     Shear = np.zeros(len(X))
     Moment = np.zeros(len(X))

     for i, x in enumerate(X):
         shear = 0 
         moment = 0

         if x>A:
             #Calculate shear and moment due to reaction at A
             shear += Va
             moment -= Va*(x-A)
        
         if x>B:
            #Calculate shear and moment due to reaction at B
             shear += Vb
             moment -= Vb*(x-B)

         if xStart< x <= xEnd:
             #Calculate shear and moment due to point load 
             shear += fy*(x-xStart)
             moment -= fy*(x-xStart)*0.5*(x-xStart)
         elif x>xEnd:
            shear += fy*(xEnd - xStart)
            moment -= fy*(xEnd - xStart)*(x-xStart-0.5*(xEnd-xStart))

         Shear[i] = shear
         Moment[i] = moment

     return Shear, Moment

#Reaction Calculation 
PL_record = np.empty([0,3])
if len(pointLoads)>0:
    for n, p in enumerate(pointLoads): 
        va, vb, ha = reactions_PL(n)  #Calculate reactions
        PL_record = np.append(PL_record, [np.array([va, ha, vb])], axis = 0)  #Store reactions for each point load

        #Adding reactions to record 
        reactions[0] += va
        reactions[1] += ha
        reactions[2] += vb 

UDL_record = np.empty([0,2])
if nUDL>0:
    for n, p in enumerate(distributedLoads): 
        va, vb = reactions_UDL(n)  #Calculate reactions
        UDL_record = np.append(UDL_record, [np.array([va, vb])], axis = 0)  #Store reactions for each point load

        #Adding reactions to record 
        reactions[0] += va
        reactions[2] += vb 

#Cycle through all point loads to determine the shear and moment
if len(pointLoads)>0:
    for n, p in enumerate(pointLoads): 
        Shear, Moment = shear_moment_PL(n)
        shearForce += Shear  # Sum up shear forces
        bendingMoment += Moment  # Sum up bending moments

#Cycle through all UDLs to determine the shear and moment
if nUDL>0:
    for n, p in enumerate(distributedLoads): 
        Shear, Moment = shear_moment_UDL(n)
        shearForce += Shear  # Sum up shear forces
        bendingMoment += Moment  # Sum up bending moments




#Plotting
one = round(reactions[0],2)
two = round(reactions[1],2)
three = round(reactions[2],2)

print(f'The vertical reaction at A is {one} kN' )
print(f'The vertical reaction at B is {three} kN')
print(f'The horizontal reaction at A is {two} kN')

# Plotting the Shear Force Diagram
layout = go.Layout(
    title={'text': 'Shear Force Diagram', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
    yaxis=dict(title='Shear Force (kN)', autorange='reversed'),
    xaxis=dict(title='Distance (m)', range=[-1, span + 1]),
    showlegend=False
)

line = go.Scatter(
    x=X,
    y=-shearForce,  # Removed negative sign
    mode='lines',
    name='Shear Force',
    fill='tonexty',
    line=dict(color='blue'),
    fillcolor='rgba(0,0,255,0.1)'
)

axis = go.Scatter(
    x=[0, span],
    y=[0, 0],
    mode='lines',
    line_color='black'
)

# Generate figure for shear force
fig = go.Figure(data=[line, axis], layout=layout)

# Display plot 
pio.show(fig)

# Plotting the Bending Moment Diagram
layout = go.Layout(
    title={'text': 'Bending Moment Diagram', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
    yaxis=dict(title='Bending Moment (kNm)'),
    xaxis=dict(title='Distance (m)', range=[-1, span + 1]),
    showlegend=False
)

line = go.Scatter(
    x=X,
    y=bendingMoment,  # Removed negative sign
    mode='lines',
    name='Bending Moment',
    fill='tonexty',
    line=dict(color='red'),
    fillcolor='rgba(255,0,0,0.1)'
)

axis = go.Scatter(
    x=[0, span],
    y=[0, 0],
    mode='lines',
    line_color='black'
)

# Generate figure for bending moment
fig = go.Figure(data=[line, axis], layout=layout)

# Display plot 
pio.show(fig)


#Calculating Deflection 

