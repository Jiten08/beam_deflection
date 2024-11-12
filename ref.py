import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Original beam parameters
span = 17
A = 3
B = 13
delta = 0.2  
noise_level = 0.05
X = np.arange(0, span + delta, delta)

# Point and distributed loads
pointLoads = np.array([[6, 0, -90]])  #[location, xMag, yMag]
distributedLoads = np.array([[8, 17, -10]])  #[StartLocation, EndLocation, Magnitude]

# Material properties
E = 1.9e11  #Young's Modulus
I = (0.2 * 0.5**3) / 12  # Moment of Inertia 
EI = E * I   

def reactions_PL(n):
    xp = pointLoads[n][0]
    fx = pointLoads[n][1]
    fy = pointLoads[n][2]
    
    la_p = A - xp
    mp = fy * la_p
    la_vb = B - A
    
    Vb = mp / la_vb
    Va = -fy - Vb
    Ha = -fx
    
    return Va, Vb, Ha

def reactions_UDL(n):
    xStart = distributedLoads[n][0]
    xEnd = distributedLoads[n][1]
    fy = distributedLoads[n][2]
    
    fy_Res = fy * (xEnd - xStart)
    x_Res = xStart + 0.5 * (xEnd - xStart)
    
    la_p = A - x_Res
    mp = fy_Res * la_p
    la_vb = B - A
    
    Vb = mp / la_vb
    Va = -fy_Res - Vb
    
    return Va, Vb

def calculate_bending_moment():
    reactions = np.array([0.0, 0, 0])
    bendingMoment = np.zeros(len(X))
    
    # Calculate reactions for point loads
    PL_record = np.empty([0, 3])
    if len(pointLoads) > 0:
        for n, p in enumerate(pointLoads):
            va, vb, ha = reactions_PL(n)
            PL_record = np.append(PL_record, [np.array([va, ha, vb])], axis=0)
            reactions[0] += va
            reactions[1] += ha
            reactions[2] += vb

    # Calculate reactions for distributed loads
    UDL_record = np.empty([0, 2])
    if len(distributedLoads[0]) > 0:
        for n, p in enumerate(distributedLoads):
            va, vb = reactions_UDL(n)
            UDL_record = np.append(UDL_record, [np.array([va, vb])], axis=0)
            reactions[0] += va
            reactions[2] += vb

    # Calculate moments from point loads
    if len(pointLoads) > 0:
        for n, p in enumerate(pointLoads):
            xp = pointLoads[n][0]
            fy = pointLoads[n][2]
            Va = PL_record[n][0]
            Vb = PL_record[n][2]

            for i, x in enumerate(X):
                moment = 0
                if x > A:
                    moment -= Va * (x - A)
                if x > B:
                    moment -= Vb * (x - B)
                if x > xp:
                    moment -= fy * (x - xp)
                bendingMoment[i] += moment

    # Calculate moments from distributed loads
    if len(distributedLoads[0]) > 0:
        for n, p in enumerate(distributedLoads):
            xStart = distributedLoads[n][0]
            xEnd = distributedLoads[n][1]
            fy = distributedLoads[n][2]
            Va = UDL_record[n][0]
            Vb = UDL_record[n][1]

            for i, x in enumerate(X):
                moment = 0
                if x > A:
                    moment -= Va * (x - A)
                if x > B:
                    moment -= Vb * (x - B)
                if xStart < x <= xEnd:
                    moment -= fy * (x - xStart) * 0.5 * (x - xStart)
                elif x > xEnd:
                    moment -= fy * (xEnd - xStart) * (x - xStart - 0.5 * (xEnd - xStart))
                bendingMoment[i] += moment

    return bendingMoment

def calculate_deflection(bendingMoment):
    supportindexA = np.where(X >= A)[0][0]
    supportindexB = np.where(X >= B)[0][0]
    
    def calc_defl(M, theta_0, v_0):
        Rotation = np.zeros(len(M))
        Rotation[supportindexA] = theta_0
        Deflection = np.zeros(len(M))
        Deflection[supportindexA] = v_0
        
        for i in range(supportindexA + 1, len(M)):
            M_avg = 0.5 * (M[i - 1] + M[i])
            Rotation[i] = Rotation[i - 1] + (M_avg / EI) * delta
            Deflection[i] = Deflection[i - 1] + 0.5 * (Rotation[i - 1] + Rotation[i]) * delta
        
        return Rotation, Deflection
    
    def findInitialRotation(M, tolerance=1e-10):
        theta_min, theta_max = -0.01, 0.01
        
        for _ in range(50):
            theta_mid = (theta_min + theta_max) / 2
            _, Deflection = calc_defl(M, theta_mid, 0)
            
            if abs(Deflection[supportindexB]) < tolerance:
                return theta_mid
            elif Deflection[supportindexB] > 0:
                theta_max = theta_mid
            else:
                theta_min = theta_mid
        
        return theta_mid
    
    initial_rotation = findInitialRotation(-bendingMoment)
    _, Deflection = calc_defl(-bendingMoment, initial_rotation, 0)
    
    return Deflection

# Calculate true deflection
bending_moment = calculate_bending_moment()
true_deflection = calculate_deflection(bending_moment)

# Add noise to deflection data
noisy_deflection = true_deflection + np.random.normal(0, np.abs(true_deflection * noise_level))

# Prepare data for ML
X_data = X.reshape(-1, 1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(noisy_deflection.reshape(-1, 1))

# Create Dash app
app = dash.Dash(__name__)

theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}
app.layout = html.Div([
    html.H1("Beam Deflection Analysis with ML"),
    dcc.Graph(id='deflection-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0
    ),
    html.Div([
        html.H3("Statistics"),
        html.Div(id='deflection-stats')
    ])
])

@app.callback(
    [Output('deflection-graph', 'figure'),
     Output('deflection-stats', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    # Train model with increasing data points
    n_points = min(len(X), 10 + n_intervals)
    
    # Create and train the model
    model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000)
    model.fit(X_scaled[:n_points], y_scaled[:n_points])
    
    # Generate predictions
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Create the figure
    fig = go.Figure()
    
    # Add measured data points
    fig.add_trace(go.Scatter(
        x=X[:n_points],
        y=noisy_deflection[:n_points],
        mode='markers',
        name='Measured Data',
        marker=dict(color='blue', size=8)
    ))
    
    # Add ML prediction
    fig.add_trace(go.Scatter(
        x=X,
        y=y_pred,
        mode='lines',
        name='ML Prediction',
        line=dict(color='red')
    ))
    
    # Add true deflection
    fig.add_trace(go.Scatter(
        x=X,
        y=true_deflection,
        mode='lines',
        name='True Deflection',
        line=dict(color='green', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Beam Deflection Analysis',
        xaxis_title='Distance (m)',
        yaxis_title='Deflection (m)',
        showlegend=True
    )
    
    # Calculate statistics
    max_measured = np.max(np.abs(noisy_deflection[:n_points]))
    max_predicted = np.max(np.abs(y_pred))
    max_true = np.max(np.abs(true_deflection))
    
    rmse = np.sqrt(np.mean((y_pred - true_deflection) ** 2))
    
    stats = html.Div([
        html.P(f"Maximum measured deflection: {max_measured:.2e} m"),
        html.P(f"Maximum predicted deflection: {max_predicted:.2e} m"),
        html.P(f"Maximum true deflection: {max_true:.2e} m"),
        html.P(f"RMSE of prediction: {rmse:.2e} m")
    ])
    
    return fig, stats

if __name__ == '__main__':
    app.run_server(debug=True)