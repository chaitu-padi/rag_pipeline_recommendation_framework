import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create figure
fig = go.Figure()

# Define colors for different layers (using the brand colors)
colors = {
    'input': '#1FB8CD',      # Strong cyan
    'evaluation': '#DB4545',  # Bright red  
    'components': '#2E8B57',  # Sea green
    'framework': '#5D878F',   # Cyan
    'output': '#D2BA4C'       # Moderate yellow
}

# Define component positions (x, y coordinates)
# Layer 1: User Input Layer (top)
input_components = [
    ('Data Sources', 1, 9),
    ('Use Case Reqs', 3, 9),
    ('Perf Reqs', 5, 9),
    ('Op Constraints', 7, 9)
]

# Layer 2: Evaluation Engine (center-top)
evaluation_engine = [
    ('Param Collection', 2, 7),
    ('Multi-Criteria', 4, 7),
    ('Optimization', 6, 7),
    ('Results Analysis', 8, 7)
]

# Layer 3: Component Evaluation Modules (middle)
component_modules = [
    ('Embedding Eval', 1, 5),
    ('Chunking Eval', 3, 5),
    ('Vector DB Eval', 5, 5),
    ('LLM Selection', 7, 5),
    ('Retrieval Eval', 2, 4),
    ('Prompt Eval', 6, 4)
]

# Layer 4: Evaluation Framework (bottom)
framework_components = [
    ('ROUGE/BLEU', 2, 2),
    ('RAGAS Metrics', 4, 2),
    ('Perf Benchmarks', 6, 2),
    ('Cost Analysis', 8, 2)
]

# Layer 5: Output Layer (bottom)
output_components = [
    ('Pipeline Config', 1, 0.5),
    ('Perf Predictions', 3, 0.5),
    ('Implementation', 5, 0.5),
    ('Monitoring', 7, 0.5)
]

# Add all components as scatter points
all_components = [
    (input_components, colors['input'], 'User Input Layer'),
    (evaluation_engine, colors['evaluation'], 'Evaluation Engine'),
    (component_modules, colors['components'], 'Component Modules'),
    (framework_components, colors['framework'], 'Framework'),
    (output_components, colors['output'], 'Output Layer')
]

for components, color, layer_name in all_components:
    x_coords = [comp[1] for comp in components]
    y_coords = [comp[2] for comp in components]
    names = [comp[0] for comp in components]
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=120,
            color=color,
            symbol='square',
            line=dict(width=2, color='white')
        ),
        text=names,
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=layer_name,
        hoverinfo='text',
        hovertext=[f'{layer_name}: {name}' for name in names]
    ))

# Add arrows showing data flow between layers
arrow_connections = [
    # From Input to Evaluation Engine
    ((2, 9), (3, 7.3)),
    ((4, 9), (5, 7.3)),
    ((6, 9), (7, 7.3)),
    # From Evaluation Engine to Component Modules
    ((3, 6.7), (2, 5.3)),
    ((5, 6.7), (4, 5.3)),
    ((7, 6.7), (6, 5.3)),
    # From Component Modules to Framework
    ((2, 4.7), (3, 2.3)),
    ((4, 4.7), (5, 2.3)),
    ((6, 4.7), (7, 2.3)),
    # From Framework to Output
    ((3, 1.7), (2, 0.8)),
    ((5, 1.7), (4, 0.8)),
    ((7, 1.7), (6, 0.8))
]

for start, end in arrow_connections:
    fig.add_annotation(
        x=end[0], y=end[1],
        ax=start[0], ay=start[1],
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='gray',
        showarrow=True
    )

# Update layout
fig.update_layout(
    title='RAG Evaluation System Architecture',
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, 9.5]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.5, 10]
    ),
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Update traces to remove cliponaxis
fig.update_traces(cliponaxis=False)

# Save the chart as PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

fig.show()