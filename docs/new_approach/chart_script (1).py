import plotly.graph_objects as go
import plotly.io as pio

# Create a simplified hierarchical flowchart
fig = go.Figure()

# Define the stages and their steps
stages = {
    "Stage 1: Data Collection": ["User Req", "Sample Data", "Baseline", "Init Perf"],
    "Stage 2: Component Eval": ["Embed Test", "Chunk Opt", "Vector DB", "LLM Eval", "Retrieval", "Prompt Opt"],
    "Stage 3: Optimization": ["Param Gen", "Multi Test", "Perf Meas", "Stats Anal", "Param Sel", "Continue?"],
    "Stage 4: Integration": ["Pipeline", "Cross Val", "Perf Valid", "Cost Anal"],
    "Stage 5: Output": ["Config Rec", "Perf Pred", "Impl Guide", "Monitor"]
}

# Colors for each stage
stage_colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
stage_names = list(stages.keys())

# Add stage headers and process boxes
for i, (stage_name, processes) in enumerate(stages.items()):
    y_pos = y_positions[i]
    color = stage_colors[i]
    
    # Add stage header
    fig.add_shape(
        type="rect",
        x0=0.05, y0=y_pos+0.05,
        x1=0.95, y1=y_pos+0.08,
        fillcolor=color,
        line=dict(color="black", width=1)
    )
    
    fig.add_annotation(
        x=0.5, y=y_pos+0.065,
        text=stage_name,
        showarrow=False,
        font=dict(size=14, color="white"),
        xanchor="center",
        yanchor="middle"
    )
    
    # Add process boxes
    box_width = 0.12
    total_width = len(processes) * box_width
    start_x = 0.5 - total_width/2
    
    for j, process in enumerate(processes):
        x_pos = start_x + j * box_width + box_width/2
        
        # Special handling for decision point
        if process == "Continue?":
            # Diamond shape for decision
            fig.add_shape(
                type="path",
                path=f"M {x_pos} {y_pos+0.025} L {x_pos+0.04} {y_pos} L {x_pos} {y_pos-0.025} L {x_pos-0.04} {y_pos} Z",
                fillcolor=color,
                line=dict(color="black", width=1)
            )
        else:
            # Rectangle for process
            fig.add_shape(
                type="rect",
                x0=x_pos-0.04, y0=y_pos-0.025,
                x1=x_pos+0.04, y1=y_pos+0.025,
                fillcolor=color,
                line=dict(color="black", width=1)
            )
        
        # Add text
        fig.add_annotation(
            x=x_pos, y=y_pos,
            text=process,
            showarrow=False,
            font=dict(size=10, color="white"),
            xanchor="center",
            yanchor="middle"
        )

# Add flow arrows between stages
arrow_positions = [
    (0.5, 0.85, 0.5, 0.75),  # Stage 1 to 2
    (0.5, 0.65, 0.5, 0.55),  # Stage 2 to 3
    (0.5, 0.45, 0.5, 0.35),  # Stage 3 to 4
    (0.5, 0.25, 0.5, 0.15)   # Stage 4 to 5
]

for x0, y0, x1, y1 in arrow_positions:
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        arrowhead=2,
        arrowsize=2,
        arrowwidth=3,
        arrowcolor="black"
    )

# Add iteration loop arrow from Stage 3 decision back to Stage 3 start
fig.add_annotation(
    x=0.2, y=0.5,
    ax=0.8, ay=0.5,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=3,
    arrowcolor="red"
)

# Add loop label
fig.add_annotation(
    x=0.05, y=0.52,
    text="Iteration Loop",
    showarrow=False,
    font=dict(size=10, color="red"),
    xanchor="left",
    yanchor="middle"
)

# Add start and end circles
fig.add_shape(
    type="circle",
    x0=0.47, y0=0.97,
    x1=0.53, y1=1.03,
    fillcolor="#13343B",
    line=dict(color="black", width=1)
)

fig.add_annotation(
    x=0.5, y=1.0,
    text="START",
    showarrow=False,
    font=dict(size=10, color="white"),
    xanchor="center",
    yanchor="middle"
)

fig.add_shape(
    type="circle",
    x0=0.47, y0=0.02,
    x1=0.53, y1=0.08,
    fillcolor="#13343B",
    line=dict(color="black", width=1)
)

fig.add_annotation(
    x=0.5, y=0.05,
    text="END",
    showarrow=False,
    font=dict(size=10, color="white"),
    xanchor="center",
    yanchor="middle"
)

# Add arrows from start to stage 1 and from stage 5 to end
fig.add_annotation(
    x=0.5, y=0.95,
    ax=0.5, ay=1.0,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=3,
    arrowcolor="black"
)

fig.add_annotation(
    x=0.5, y=0.08,
    ax=0.5, ay=0.05,
    arrowhead=2,
    arrowsize=2,
    arrowwidth=3,
    arrowcolor="black"
)

# Update layout
fig.update_layout(
    title="RAG Evaluation Process Flow",
    showlegend=False,
    xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    font=dict(size=12)
)

# Save as both PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")

fig.show()