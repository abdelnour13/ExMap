"""
Visualisation utilities.

Builds interactive graph visualisations using ``neo4j-viz`` (the official
Neo4j Python visualisation library, wrapping the NVL JavaScript library)
and writes self-contained HTML files under /public/visualizations/.

neo4j-viz API summary used here:
  - Node(id, caption, size, color, properties)
  - Relationship(source, target, caption, color, width)
  - VisualizationGraph(nodes, relationships)
  - vg.render(layout, height, theme) → IPython.HTML  (.data = raw HTML string)
"""

import uuid
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from neo4j_viz import Node, Relationship, VisualizationGraph, Layout
from api.constants import VIS_DIR

VIS_DIR = Path(VIS_DIR)


# ── Colour palette (hex strings, one per node type) ───────────────────────────

_COLOURS: dict[str, str] = {
    "Domain":       "#534AB7",   # purple
    "SubDomain":    "#1D9E75",   # teal
    "Expertise":    "#D85A30",   # coral
    "Entity":       "#888780",   # gray
    "Article":      "#378ADD",   # blue
    "Lab":          "#378ADD",
    "Venue":        "#378ADD",
    "ExpertiseMap": "#7F77DD",
}

# Node sizes (radius in pixels) by type
_SIZES: dict[str, int] = {
    "Domain":      35,
    "SubDomain":   30,
    "Expertise":   28,
    "Entity":      20,
    "Article":     25,
    "Lab":         25,
    "Venue":       25,
    "ExpertiseMap": 40,
}

# Relationship colours by relation type
_REL_COLOURS: dict[str, str] = {
    "ISSUBDOMAINOF":  "#AFA9EC",   # light purple
    "BELONGSTO":      "#9FE1CB",   # light teal
    "HASRELATIONWITH": "#D3D1C7",  # light gray
    "APPEARSIN":      "#B5D4F4",   # light blue
}

_DEFAULT_REL_COLOUR = "#D3D1C7"
_DEFAULT_NODE_COLOUR = "#B4B2A9"


def _node_colour(node_type: str) -> str:
    return _COLOURS.get(node_type, _DEFAULT_NODE_COLOUR)


def _rel_colour(rel_type: str) -> str:
    return _REL_COLOURS.get(rel_type, _DEFAULT_REL_COLOUR)


# ── File helpers ──────────────────────────────────────────────────────────────

class VisualizationService:

    @staticmethod
    def _new_filename() -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"{ts}_{uid}.html"

    def _render_and_save(self, 
        vg: VisualizationGraph, 
        layout: Layout = Layout.FORCE_DIRECTED
    ) -> str:
        html_obj = vg.render(layout=layout, height="750px")
        filename = self._new_filename()
        (VIS_DIR / filename).write_text(html_obj.data, encoding="utf-8")
        return filename


    def build_vis(self, 
        data : list[dict], 
        layout : Layout = Layout.FORCE_DIRECTED
    ) -> str:

        nodes: dict[str, Node] = {}
        relationships: dict[str, Node] = {}

        def _node_id(node : dict) -> str:
            return f"{node['labels'][0]}_{node['id']}"

        degrees = Counter(
            [_node_id(row['src']) for row in data] + [_node_id(row['dst']) for row in data]
        )

        def _node_size(node: dict) -> int:

            size = _SIZES.get(node['labels'][0], 16)

            if 'Entity' in node['labels']:
                size = size + 10 * (degrees.get(_node_id(node)) / degrees.most_common(1)[0][1])

        def _caption(node : dict) -> str:

            if 'llm_label' in node and node['llm_label'] is not None and 'Expertise' in node['labels']:
                return node['llm_label']

            if 'personalized_pagerank_name' in node and node['personalized_pagerank_name'] is not None:
                return node['personalized_pagerank_name']
            
            if 'name' in node and node['name'] is not None:
                return node['name']
            
            return node['labels'][0]

        def _rel_caption(row : dict) -> str:

            props = row['props']

            if props.get('predicate') is not None:
                return props.get('predicate')
            
            if props.get('score') is not None:
                return f'{row["relation"]} : {100 * props.get("score"):.2f}%'
            
            return row['relation']
        
        def _node_props(d : dict) -> dict:

            r = {
                'name' : d.get('name'),
                'llm_label': d.get('llm_label'),
            } 

            return {
                key : value
                for key, value in r.items() 
                if value is not None
            }
        
        def _rel_props(d : dict) -> dict:

            r = {
                'predicate' : d.get('predicate'),
                'score': d.get('score'),
            } 

            return {
                key : value
                for key, value in r.items() 
                if value is not None
            }

        for row in data:
            
            src = row['src']
            dst = row['dst']
            relation = row['relation']

            src_id = _node_id(src)
            dst_id = _node_id(dst)
            rel_id = f"{src_id}_{relation}_{dst_id}"

            nodes[src_id] = Node(
                id=src_id,
                caption=_caption(src),
                size=_node_size(src),
                color=_node_colour(src['labels'][0]),
                properties=_node_props(src)
            )

            nodes[dst_id] = Node(
                id=dst_id,
                caption=_caption(dst),
                size=_node_size(dst),
                color=_node_colour(dst['labels'][0]),
                properties=_node_props(dst)
            )

            relationships[rel_id] = Relationship(
                source=src_id,
                target=dst_id,
                caption=_rel_caption(row),
                color=_rel_colour(relation),
                properties=_rel_props(row['props'])
            )

        nodes = list(nodes.values())
        relationships = list(relationships.values())

        vg = VisualizationGraph(nodes=nodes, relationships=relationships)
        return self._render_and_save(vg, layout=layout)


    def build_taxonomy_vis(self, data: list[dict]) -> str:
        return self.build_vis(data, Layout.HIERARCHICAL)

    def build_expertises_vis(self, data: list[dict]) -> str:
        return self.build_vis(data)

    def build_expertise_detail_vis(self, data: list[dict]) -> str:
        return self.build_vis(data)

    @staticmethod
    def _pie_fig(
        labels: list[str],
        values: list[int],
        title: str,
    ) -> go.Figure:
        
        if not labels:

            fig = go.Figure()

            fig.add_annotation(
                text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font={"size": 14, "color": "#aaa"},
            )

            fig.update_layout(title=title, height=320, paper_bgcolor="#ffffff")

            return fig
 
        n_colors = len(labels)
        colours = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
 
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.38,
            marker_colors=colours,
            textinfo="percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            sort=True,
        ))

        fig.update_layout(
            title={"text": title, "font": {"size": 13, "color": "#1c1917"}, "x": 0},
            height=320,
            margin={"l": 10, "r": 10, "t": 44, "b": 10},
            paper_bgcolor="#ffffff",
            font={"family": "system-ui, sans-serif", "color": "#1c1917"},
            legend={"font": {"size": 9}, "orientation": "v"},
            showlegend=True,
        )

        return fig

    @staticmethod
    def _grid_pie_fig(
        rows: list[dict],
        group_key: str,
        stack_key: str,
        value_key: str,
        title: str,
    ) -> go.Figure:
    
        if not rows:

            fig = go.Figure()

            fig.add_annotation(
                text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font={"size": 14, "color": "#aaa"},
            )

            fig.update_layout(title=title, height=320, paper_bgcolor="#ffffff")

            return fig
    
        df = pd.DataFrame(rows)
        labs = sorted(df[stack_key].unique().tolist())
        groups_all = sorted(df[group_key].unique().tolist())
    
        # Consistent colour per group across all pies
        n_colors = len(groups_all)
        colours = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])
        group_colours = {
            g: colours[i]
            for i, g in enumerate(groups_all)
        }
    
        n_cols = min(len(labs), 4)          # max 4 pies per row
        n_rows = -(-len(labs) // n_cols)    # ceiling division
    
        specs = [[{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)]
        subplot_titles = labs + [""] * (n_rows * n_cols - len(labs))
    
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=specs,
            subplot_titles=subplot_titles,
        )
    
        for idx, lab in enumerate(labs):

            r = idx // n_cols + 1
            c = idx %  n_cols + 1
            lab_df  = df[df[stack_key] == lab]
            glabels = lab_df[group_key].tolist()
            gvalues = lab_df[value_key].tolist()
            colours = [group_colours[g] for g in glabels]
    
            fig.add_trace(
                go.Pie(
                    labels=glabels,
                    values=gvalues,
                    hole=0.38,
                    marker_colors=colours,
                    textinfo="percent",
                    hovertemplate=f"{lab}<br>%{{label}}: %{{value}} (%{{percent}})<extra></extra>",
                    showlegend=True,   # legend entries from first pie only
                    sort=True,
                    name=lab,
                ),
                row=r, col=c,
            )
    
        for ann in fig.layout.annotations:
            ann.font.size = 10
    
        fig.update_layout(
            title={"text": title, "font": {"size": 13, "color": "#1c1917"}, "x": 0},
            height=max(280, n_rows * 280),
            margin={"l": 10, "r": 10, "t": 56, "b": 10},
            paper_bgcolor="#ffffff",
            font={"family": "system-ui, sans-serif", "color": "#1c1917"},
            legend={"font": {"size": 9}, "orientation": "v", "title": {"text": group_key.replace("_", " ").title(), "font": {"size": 9}}},
        )

        return fig

 
    def build_summary_vis(self, data: dict) -> str:
    
        STACK_PALETTE = [
            "#534AB7", "#1D9E75", "#D85A30", "#378ADD",
            "#BA7517", "#0F6E56", "#993C1D", "#185FA5",
            "#3B6D11", "#712B13", "#085041", "#3C3489",
        ]
    
        PLOTLY_CONFIG = {
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "responsive": True,
        }
        
        # ── Build all 10 figures ──────────────────────────────────────────────────
    
        figs: list[tuple[str, go.Figure, bool]] = []  # (section, fig, is_wide)
    
        # — Articles —
        figs.append(("articles", self._pie_fig(
            [r["lab"] for r in data["articles_by_lab"]],
            [r["numArticles"] for r in data["articles_by_lab"]],
            "Articles by Lab",
        ), False))
    
        figs.append(("articles", self._pie_fig(
            [r["domain"] for r in data["articles_by_domain"]],
            [r["numArticles"] for r in data["articles_by_domain"]],
            "Articles by Domain",
        ), False))
    
        figs.append(("articles", self._pie_fig(
            [r["sub_domain"] for r in data["articles_by_subdomain"]],
            [r["numArticles"] for r in data["articles_by_subdomain"]],
            "Articles by SubDomain",
        ), False))
    
        figs.append(("articles", self._grid_pie_fig(
            data["articles_by_domain_lab"],
            group_key="domain", stack_key="lab", value_key="numArticles",
            title="Articles by Domain & Lab",
        ), True))
    
        figs.append(("articles", self._grid_pie_fig(
            data["articles_by_subdomain_lab"],
            group_key="sub_domain", stack_key="lab", value_key="numArticles",
            title="Articles by SubDomain & Lab",
        ), True))
    
        # — Expertises —
        figs.append(("expertises", self._pie_fig(
            [r["lab"] for r in data["expertises_by_lab"]],
            [r["numExpertises"] for r in data["expertises_by_lab"]],
            "Expertises by Lab",
        ), False))
    
        figs.append(("expertises", self._pie_fig(
            [r["domain"] for r in data["expertises_by_domain"]],
            [r["numExpertises"] for r in data["expertises_by_domain"]],
            "Expertises by Domain",
        ), False))
    
        figs.append(("expertises", self._pie_fig(
            [r["sub_domain"] for r in data["expertises_by_subdomain"]],
            [r["numExpertises"] for r in data["expertises_by_subdomain"]],
            "Expertises by SubDomain",
        ), False))
    
        figs.append(("expertises", self._grid_pie_fig(
            data["expertises_by_domain_lab"],
            group_key="domain", stack_key="lab", value_key="numExpertises",
            title="Expertises by Domain & Lab",
        ), True))
    
        figs.append(("expertises", self._grid_pie_fig(
            data["expertises_by_subdomain_lab"],
            group_key="sub_domain", stack_key="lab", value_key="numExpertises",
            title="Expertises by SubDomain & Lab",
        ), True))
        
        chart_divs: list[str] = []

        for i, (_, fig, wide) in enumerate(figs):

            include_js = (i == 0) 

            div_html = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=include_js,
                config=PLOTLY_CONFIG,
                div_id=f"chart-{i}",
            )

            css_class = "card wide" if wide else "card"
            chart_divs.append(f'<div class="{css_class}">{div_html}</div>')
    
        art_narrow       = "\n".join(chart_divs[0:3])  # 3 simple
        domain_lab_row   = "\n".join([chart_divs[3], chart_divs[8]])  # side by side
        art_sd_lab       = chart_divs[4]                # full-width stacked
        exp_narrow       = "\n".join(chart_divs[5:8])  # 3 simple
        exp_sd_lab       = chart_divs[9]                # full-width stacked
    
        html = ""
    
        filename = self._new_filename()
        (VIS_DIR / filename).write_text(html, encoding="utf-8")
        return filename