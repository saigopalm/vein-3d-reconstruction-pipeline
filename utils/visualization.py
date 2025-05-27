import plotly.graph_objects as go

def plot_3d(p3ds):
    fig = go.Figure(data=go.Scatter3d(
        x=p3ds[:, 0],
        y=p3ds[:, 1],
        z=p3ds[:, 2],
        mode='markers',
        marker=dict(size=2, color=p3ds[:, 2], colorscale='Viridis', opacity=0.8)
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title="3D Veins Visualization"
    )
    fig.show()