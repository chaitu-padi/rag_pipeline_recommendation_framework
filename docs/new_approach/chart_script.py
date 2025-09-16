import plotly.graph_objects as go
import plotly.io as pio
import math

# Create a comprehensive flowchart diagram for the Evaluation Logic and Algorithms
fig = go.Figure()

# Define colors for different component types
colors = {
    'engine': '#1FB8CD',  # Strong cyan for engines
    'algorithm': '#DB4545',  # Bright red for algorithms  
    'calculator': '#2E8B57',  # Sea green for calculators
    'optimization': '#5D878F',  # Cyan for optimization
    'analysis': '#D2BA4C'  # Moderate yellow for analysis
}

# Define components with better alignment and closer spacing
components = [
    # Multi-Criteria Decision Engine (rectangles) - perfectly aligned
    {'name': 'Param Space\nGen', 'x': 1, 'y': 7, 'type': 'engine', 'shape': 'rect'},
    {'name': 'Constraint\nSat Logic', 'x': 2.2, 'y': 7, 'type': 'engine', 'shape': 'rect'},
    {'name': 'Pareto Front\nCalc', 'x': 3.4, 'y': 7, 'type': 'engine', 'shape': 'rect'},
    {'name': 'Weight\nScoring', 'x': 4.6, 'y': 7, 'type': 'engine', 'shape': 'rect'},
    
    # Component Evaluation Algorithms (circles) - aligned
    {'name': 'Embed\nBenchmark', 'x': 0.4, 'y': 5.8, 'type': 'algorithm', 'shape': 'circle'},
    {'name': 'Chunk Test\nAlgorithm', 'x': 1.6, 'y': 5.8, 'type': 'algorithm', 'shape': 'circle'},
    {'name': 'Vector DB\nPerf Test', 'x': 2.8, 'y': 5.8, 'type': 'algorithm', 'shape': 'circle'},
    {'name': 'LLM Quality\nAssessment', 'x': 4, 'y': 5.8, 'type': 'algorithm', 'shape': 'circle'},
    {'name': 'Retrieval\nEffectiveness', 'x': 5.2, 'y': 5.8, 'type': 'algorithm', 'shape': 'circle'},
    
    # Metrics Calculation Engines (diamonds) - aligned
    {'name': 'ROUGE\nCalculator', 'x': 1, 'y': 4.6, 'type': 'calculator', 'shape': 'diamond'},
    {'name': 'BLEU\nCalculator', 'x': 2.2, 'y': 4.6, 'type': 'calculator', 'shape': 'diamond'},
    {'name': 'RAGAS\nEngine', 'x': 3.4, 'y': 4.6, 'type': 'calculator', 'shape': 'diamond'},
    {'name': 'Performance\nMetrics', 'x': 4.6, 'y': 4.6, 'type': 'calculator', 'shape': 'diamond'},
    
    # Optimization Algorithms (hexagons) - aligned
    {'name': 'Bayesian\nOptimization', 'x': 0.4, 'y': 3.4, 'type': 'optimization', 'shape': 'hexagon'},
    {'name': 'Grid Search\nImplement', 'x': 1.6, 'y': 3.4, 'type': 'optimization', 'shape': 'hexagon'},
    {'name': 'Random\nSearch', 'x': 2.8, 'y': 3.4, 'type': 'optimization', 'shape': 'hexagon'},
    {'name': 'Genetic\nAlgorithm', 'x': 4, 'y': 3.4, 'type': 'optimization', 'shape': 'hexagon'},
    
    # Statistical Analysis Module (ellipses) - aligned
    {'name': 'Significance\nTesting', 'x': 1, 'y': 2.2, 'type': 'analysis', 'shape': 'ellipse'},
    {'name': 'Confidence\nInterval', 'x': 2.2, 'y': 2.2, 'type': 'analysis', 'shape': 'ellipse'},
    {'name': 'Effect Size\nComputation', 'x': 3.4, 'y': 2.2, 'type': 'analysis', 'shape': 'ellipse'},
    {'name': 'Cross-Valid\nLogic', 'x': 4.6, 'y': 2.2, 'type': 'analysis', 'shape': 'ellipse'},
]

# Function to add different shapes with better sizing
def add_shape_and_text(fig, comp):
    x, y = comp['x'], comp['y']
    color = colors[comp['type']]
    
    if comp['shape'] == 'rect':
        fig.add_shape(
            type="rect",
            x0=x-0.4, y0=y-0.25,
            x1=x+0.4, y1=y+0.25,
            fillcolor=color,
            opacity=0.8,
            line=dict(color="black", width=2)
        )
    elif comp['shape'] == 'circle':
        fig.add_shape(
            type="circle",
            x0=x-0.35, y0=y-0.25,
            x1=x+0.35, y1=y+0.25,
            fillcolor=color,
            opacity=0.8,
            line=dict(color="black", width=2)
        )
    elif comp['shape'] == 'diamond':
        # Create diamond using path
        path = f"M {x},{y-0.25} L {x+0.4},{y} L {x},{y+0.25} L {x-0.4},{y} Z"
        fig.add_shape(
            type="path",
            path=path,
            fillcolor=color,
            opacity=0.8,
            line=dict(color="black", width=2)
        )
    elif comp['shape'] == 'hexagon':
        # Create hexagon using path
        angles = [i * 60 for i in range(6)]
        points = []
        for angle in angles:
            rad = math.radians(angle)
            px = x + 0.35 * math.cos(rad)
            py = y + 0.2 * math.sin(rad)
            points.append(f"{px},{py}")
        path = f"M {points[0]} " + " ".join([f"L {p}" for p in points[1:]]) + " Z"
        fig.add_shape(
            type="path",
            path=path,
            fillcolor=color,
            opacity=0.8,
            line=dict(color="black", width=2)
        )
    elif comp['shape'] == 'ellipse':
        fig.add_shape(
            type="circle",
            x0=x-0.45, y0=y-0.2,
            x1=x+0.45, y1=y+0.2,
            fillcolor=color,
            opacity=0.8,
            line=dict(color="black", width=2)
        )
    
    # Add text with larger font
    fig.add_annotation(
        x=x, y=y,
        text=comp['name'],
        showarrow=False,
        font=dict(size=10, color="white", family="Arial Bold"),
        xanchor="center",
        yanchor="middle"
    )

# Add all component shapes
for comp in components:
    add_shape_and_text(fig, comp)

# Add comprehensive and visible data flow arrows
arrows = [
    # Main vertical flow - Decision Engine to Component Evaluation
    {'start': (1, 6.75), 'end': (0.4, 6.05), 'color': 'black', 'width': 3},
    {'start': (2.2, 6.75), 'end': (1.6, 6.05), 'color': 'black', 'width': 3},
    {'start': (3.4, 6.75), 'end': (2.8, 6.05), 'color': 'black', 'width': 3},
    {'start': (4.6, 6.75), 'end': (4, 6.05), 'color': 'black', 'width': 3},
    
    # Component Evaluation to Metrics
    {'start': (1.6, 5.55), 'end': (1, 4.85), 'color': 'black', 'width': 3},
    {'start': (1.6, 5.55), 'end': (2.2, 4.85), 'color': 'black', 'width': 3},
    {'start': (2.8, 5.55), 'end': (3.4, 4.85), 'color': 'black', 'width': 3},
    {'start': (4, 5.55), 'end': (4.6, 4.85), 'color': 'black', 'width': 3},
    
    # Metrics to Optimization
    {'start': (1, 4.35), 'end': (0.4, 3.65), 'color': 'black', 'width': 3},
    {'start': (2.2, 4.35), 'end': (1.6, 3.65), 'color': 'black', 'width': 3},
    {'start': (3.4, 4.35), 'end': (2.8, 3.65), 'color': 'black', 'width': 3},
    {'start': (4.6, 4.35), 'end': (4, 3.65), 'color': 'black', 'width': 3},
    
    # Optimization to Statistical Analysis
    {'start': (0.4, 3.15), 'end': (1, 2.45), 'color': 'black', 'width': 3},
    {'start': (1.6, 3.15), 'end': (2.2, 2.45), 'color': 'black', 'width': 3},
    {'start': (2.8, 3.15), 'end': (3.4, 2.45), 'color': 'black', 'width': 3},
    {'start': (4, 3.15), 'end': (4.6, 2.45), 'color': 'black', 'width': 3},
    
    # Feedback loops - more prominent
    {'start': (0.4, 3.15), 'end': (1, 6.75), 'color': '#DB4545', 'width': 4},
    {'start': (4.6, 1.95), 'end': (4.6, 6.75), 'color': '#DB4545', 'width': 4},
    
    # Horizontal connections within layers
    {'start': (1.4, 7), 'end': (1.8, 7), 'color': 'gray', 'width': 2},
    {'start': (2.6, 7), 'end': (3.0, 7), 'color': 'gray', 'width': 2},
    {'start': (3.8, 7), 'end': (4.2, 7), 'color': 'gray', 'width': 2},
]

for arrow in arrows:
    fig.add_annotation(
        x=arrow['end'][0], y=arrow['end'][1],
        ax=arrow['start'][0], ay=arrow['start'][1],
        arrowhead=3,
        arrowsize=2,
        arrowwidth=arrow['width'],
        arrowcolor=arrow['color'],
        opacity=0.8
    )

# Add section labels with better positioning
section_labels = [
    {'text': 'Multi-Criteria Decision Engine', 'x': 2.8, 'y': 7.6, 'size': 13},
    {'text': 'Component Evaluation Algorithms', 'x': 2.8, 'y': 6.4, 'size': 13},
    {'text': 'Metrics Calculation Engines', 'x': 2.8, 'y': 5.2, 'size': 13},
    {'text': 'Optimization Algorithms', 'x': 2.8, 'y': 4.0, 'size': 13},
    {'text': 'Statistical Analysis Module', 'x': 2.8, 'y': 1.6, 'size': 13},
]

for label in section_labels:
    fig.add_annotation(
        x=label['x'], y=label['y'],
        text=label['text'],
        showarrow=False,
        font=dict(size=label['size'], color="black", family="Arial Black"),
        xanchor="center",
        yanchor="middle",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

# Add mathematical formulas with better positioning and larger text
formulas = [
    {'text': 'ROUGE-L: F=2PR/(P+R)<br>P=LCS/m, R=LCS/n', 'x': 0.7, 'y': 4.2},
    {'text': 'BLEU: BP×exp(Σwₙlogpₙ)<br>BP=min(1,e^(1-r/c))', 'x': 1.9, 'y': 4.2},
    {'text': 'RAGAS: Faith×Prec×Recall<br>Context Precision', 'x': 3.1, 'y': 4.2},
    {'text': 'Latency, Throughput<br>Memory Usage', 'x': 4.3, 'y': 4.2},
    {'text': 'Pareto: min f₁(x), f₂(x)<br>x∈feasible space', 'x': 3.1, 'y': 6.6},
    {'text': 'GP: μ(x)±σ(x)<br>Acquisition: UCB', 'x': 0.1, 'y': 3.0},
    {'text': 'Grid: x₁×x₂×...×xₙ<br>Exhaustive search', 'x': 1.3, 'y': 3.0},
]

for formula in formulas:
    fig.add_annotation(
        x=formula['x'], y=formula['y'],
        text=formula['text'],
        showarrow=False,
        font=dict(size=10, color="darkblue", family="Arial"),
        bgcolor="lightyellow",
        bordercolor="gray",
        borderwidth=1,
        opacity=0.95
    )

# Add legend closer to the diagram
legend_items = [
    {'shape': 'rect', 'color': '#1FB8CD', 'label': 'Engines', 'x': 5.8, 'y': 6.8},
    {'shape': 'circle', 'color': '#DB4545', 'label': 'Algorithms', 'x': 5.8, 'y': 6.4},
    {'shape': 'diamond', 'color': '#2E8B57', 'label': 'Calculators', 'x': 5.8, 'y': 6.0},
    {'shape': 'hexagon', 'color': '#5D878F', 'label': 'Optimization', 'x': 5.8, 'y': 5.6},
    {'shape': 'ellipse', 'color': '#D2BA4C', 'label': 'Analysis', 'x': 5.8, 'y': 5.2},
]

# Add legend title
fig.add_annotation(
    x=5.8, y=7.3,
    text="Component Types",
    showarrow=False,
    font=dict(size=13, color="black", family="Arial Bold"),
    xanchor="center",
    bgcolor="lightgray",
    bordercolor="black",
    borderwidth=1
)

for item in legend_items:
    x, y = item['x'], item['y']
    
    # Add legend shape - larger for visibility
    if item['shape'] == 'rect':
        fig.add_shape(type="rect", x0=x-0.2, y0=y-0.1, x1=x+0.1, y1=y+0.1,
                     fillcolor=item['color'], opacity=0.8, line=dict(color="black", width=1))
    elif item['shape'] == 'circle':
        fig.add_shape(type="circle", x0=x-0.15, y0=y-0.1, x1=x+0.15, y1=y+0.1,
                     fillcolor=item['color'], opacity=0.8, line=dict(color="black", width=1))
    elif item['shape'] == 'diamond':
        path = f"M {x-0.05},{y-0.1} L {x+0.2},{y} L {x-0.05},{y+0.1} L {x-0.3},{y} Z"
        fig.add_shape(type="path", path=path, fillcolor=item['color'], opacity=0.8,
                     line=dict(color="black", width=1))
    elif item['shape'] == 'hexagon':
        angles = [i * 60 for i in range(6)]
        points = []
        for angle in angles:
            rad = math.radians(angle)
            px = x-0.05 + 0.2 * math.cos(rad)
            py = y + 0.08 * math.sin(rad)
            points.append(f"{px},{py}")
        path = f"M {points[0]} " + " ".join([f"L {p}" for p in points[1:]]) + " Z"
        fig.add_shape(type="path", path=path, fillcolor=item['color'], opacity=0.8,
                     line=dict(color="black", width=1))
    elif item['shape'] == 'ellipse':
        fig.add_shape(type="circle", x0=x-0.2, y0=y-0.08, x1=x+0.2, y1=y+0.08,
                     fillcolor=item['color'], opacity=0.8, line=dict(color="black", width=1))
    
    # Add legend text
    fig.add_annotation(
        x=x+0.4, y=y,
        text=item['label'],
        showarrow=False,
        font=dict(size=11, color="black", family="Arial"),
        xanchor="left",
        yanchor="middle"
    )

# Add flow direction indicators
fig.add_annotation(
    x=5.8, y=4.6,
    text="Data Flow",
    showarrow=False,
    font=dict(size=12, color="black", family="Arial Bold"),
    xanchor="center",
    bgcolor="lightblue",
    bordercolor="black",
    borderwidth=1
)

flow_legend = [
    {'color': 'black', 'label': 'Forward Flow', 'y': 4.2},
    {'color': '#DB4545', 'label': 'Feedback', 'y': 3.9},
    {'color': 'gray', 'label': 'Lateral Connect', 'y': 3.6},
]

for flow in flow_legend:
    fig.add_annotation(
        x=5.8, y=flow['y'],
        ax=5.4, ay=flow['y'],
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=flow['color']
    )
    fig.add_annotation(
        x=6.2, y=flow['y'],
        text=flow['label'],
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor="left",
        yanchor="middle"
    )

# Update layout with tighter bounds
fig.update_layout(
    title="Eval Logic & Algorithms Design",
    showlegend=False,
    xaxis=dict(
        range=[-0.2, 7.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[1.2, 8],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor="white"
)

# Save the chart
fig.write_image("eval_logic_design.png")
fig.write_image("eval_logic_design.svg", format="svg")

fig.show()