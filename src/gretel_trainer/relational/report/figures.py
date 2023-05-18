import math
from typing import Optional

import plotly.graph_objects as go

_GRETEL_PALETTE = ["#A051FA", "#18E7AA"]

SCORE_VALUES = [
    {"label": "Very poor", "color": "rgb(229, 60, 26)"},
    {"label": "Poor", "color": "rgb(229, 128, 26)"},
    {"label": "Average", "color": "rgb(229, 161, 26)"},
    {"label": "Good", "color": "rgb(183, 210, 45)"},
    {"label": "Excellent", "color": "rgb(72, 210, 45)"},
]

PRIVACY_LEVEL_VALUES = [
    {"label": "Poor", "color": "rgb(203, 210, 252)"},
    {"label": "Normal", "color": "rgb(160, 171, 245)"},
    {"label": "Good", "color": "rgb(124, 135, 233)"},
    {"label": "Very Good", "color": "rgb(83, 81, 222)"},
    {"label": "Excellent", "color": "rgb(59, 46, 208)"},
]


def _generate_pointer_path(score: int):
    """
    Helper to generate an svg path for the needle in the gauge and needle chart.  The path is a triangle,
    basically a tall skinny pyramid with the base at the center of the circle and the apex at the score
    on the outer ring.

    Args:
        score: Integer score in [0,100].  Pointer path will point at this value on the gauge.

    Returns:
        A string containing the raw svg path.  It does NOT return any <svg> tags.

    """
    theta = score * (282 - 34) / 100 - 34
    rads = math.radians(theta)

    radius = 0.45
    size = 0.025

    x1 = -1 * radius * math.cos(rads) + 0.5
    y1 = radius * math.sin(rads) + 0.5
    return f"""
    M {x1} {y1}
    L {-1 * size * math.cos(math.radians(theta - 90)) + 0.5}
        {size * math.sin(math.radians(theta - 90)) + 0.5}
    L {-1 * size * math.cos(math.radians(theta + 90)) + 0.5}
        {size * math.sin(math.radians(theta + 90)) + 0.5}
    Z"""


def gauge_and_needle_chart(
    score: Optional[int],
    display_score: bool = True,
    marker_colors: Optional[list[str]] = None,
) -> go.Figure:
    """
    The 'fancy' gauge and needle chart to go with the overall score of the report.  Has colored segments for
    each grade range and an svg pointer supplied by _generate_pointer_path

    Args:
        score: Integer score in [0,100].  Pointer path will point at this value on the gauge.

    Returns:
        A plotly.graph_objects.Figure

    """
    if score is None:
        fig = go.Figure(
            layout=go.Layout(
                annotations=[
                    go.layout.Annotation(
                        text="N/A",
                        font=dict(color="rgba(174, 95, 5, 1)", size=18),
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                    )
                ]
            )
        )
        marker_colors = ["rgb(220, 220, 220)", "rgba(255, 255, 255, 0)"]
        pie_values = [70, 30]
    else:
        if not marker_colors:
            marker_colors = [s["color"] for s in SCORE_VALUES]
        if marker_colors[-1] != "rgba(255, 255, 255, 0)":
            marker_colors.append("rgba(255, 255, 255, 0)")
        pie_values = [70 // (len(marker_colors) - 1)] * (len(marker_colors) - 1)
        pie_values.append(30)
        fig = go.Figure()

    fig.update_layout(
        autosize=False,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=180,
        width=180,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode=False,
        modebar=None,
    )
    fig.add_trace(
        go.Pie(
            name="gauge",
            values=pie_values,
            marker=dict(
                colors=marker_colors,
                line=dict(width=4, color="#fafafa"),
            ),
            hole=0.75,
            direction="clockwise",
            sort=False,
            rotation=234,
            showlegend=False,
            hoverinfo="none",
            textinfo="none",
            textposition="outside",
        )
    )

    if score is not None:
        if display_score:
            fig.add_trace(
                go.Indicator(
                    mode="number", value=score, domain=dict(x=[0, 1], y=[0.28, 0.45])
                )
            )
        fig.add_shape(
            type="circle", fillcolor="black", x0=0.475, x1=0.525, y0=0.475, y1=0.525
        )
        fig.add_shape(
            type="path",
            fillcolor="black",
            line=dict(width=0),
            path=_generate_pointer_path(score),
        )

    return fig
